using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

namespace DsxCollector
{
    // These fixtures bind ONLY loopback with an OS-selected ephemeral port.
    // They never discover, start, or connect to Olympus software or real devices.
    public static class RdkSelfTests
    {
        private sealed class Fixture : IDisposable
        {
            private readonly TcpListener listener;
            private TcpClient peer;
            public readonly List<string> Commands = new List<string>();
            public readonly List<string> Events = new List<string>();
            public readonly RdkClient Client;
            public readonly int Port;
            public readonly Task Server;
            public Exception ServerFailure;

            public Fixture(Func<Fixture, NetworkStream, Task> script)
            {
                listener = new TcpListener(IPAddress.Loopback, 0);
                listener.Start();
                Port = ((IPEndPoint)listener.LocalEndpoint).Port;
                Client = new RdkClient(delegate(string source, string kind, object data, string requestId, string captureId)
                {
                    lock (Events) Events.Add(kind + "|" + data + "|" + requestId + "|" + captureId);
                });
                Server = Task.Run(async delegate
                {
                    try
                    {
                        peer = await listener.AcceptTcpClientAsync().ConfigureAwait(false);
                        peer.NoDelay = true;
                        await script(this, peer.GetStream()).ConfigureAwait(false);
                    }
                    catch (Exception error) { ServerFailure = error; }
                });
            }

            public Task Connect() { return Client.ConnectAsync("127.0.0.1", Port, 3000); }

            public async Task<string> Read(NetworkStream stream)
            {
                StringBuilder text = new StringBuilder();
                byte[] one = new byte[1];
                while (text.Length < 4096)
                {
                    int count = await stream.ReadAsync(one, 0, 1).ConfigureAwait(false);
                    if (count == 0) return null;
                    text.Append((char)one[0]);
                    if (text.Length >= 2 && text[text.Length - 2] == '\r' && text[text.Length - 1] == '\n')
                    {
                        string command = text.ToString(0, text.Length - 2);
                        lock (Commands) Commands.Add(command);
                        return command;
                    }
                }
                throw new InvalidDataException("Fixture received oversized command.");
            }

            public static async Task Write(NetworkStream stream, string value)
            {
                byte[] data = Encoding.UTF8.GetBytes(value);
                await stream.WriteAsync(data, 0, data.Length).ConfigureAwait(false);
            }

            public bool HasEvent(string fragment)
            {
                lock (Events) return Events.Exists(delegate(string entry) { return entry.Contains(fragment); });
            }

            public void Dispose()
            {
                Client.Dispose();
                if (peer != null) peer.Close();
                listener.Stop();
                try { Server.Wait(1000); } catch (AggregateException) { }
            }
        }

        private static void Assert(bool condition, string message)
        {
            if (!condition) throw new InvalidOperationException(message);
        }

        private static async Task ExpectFailure(Task task, string label)
        {
            try { await task.ConfigureAwait(false); }
            catch (Exception) { return; }
            throw new InvalidOperationException("Expected rejection: " + label);
        }

        private static async Task WaitFor(Func<bool> condition, string label)
        {
            for (int i = 0; i < 100; i++)
            {
                if (condition()) return;
                await Task.Delay(20).ConfigureAwait(false);
            }
            throw new TimeoutException(label);
        }

        private static void Case(List<object> results, string name, Func<Task> test)
        {
            try
            {
                Task.Run(test).GetAwaiter().GetResult();
                results.Add(new { name = name, passed = true, detail = "Loopback fixture only; no device acceptance inferred." });
            }
            catch (Exception error)
            {
                results.Add(new { name = name, passed = false, detail = error.GetType().Name + ": " + error.Message });
            }
        }

        public static List<object> Run(string root)
        {
            // root belongs to the outer evidence run. No vendor or repo paths are written here.
            List<object> results = new List<object>();

            Case(results, "rdk_default_construct_is_passive", async delegate
            {
                int events = 0;
                using (RdkClient client = new RdkClient(delegate(string s, string k, object d, string r, string c) { events++; }))
                {
                    Assert(!client.Connected && !client.Busy && events == 0, "Construction must not connect or log a request.");
                    await ExpectFailure(client.SendAsync("ZPOS?", 500, "passive"), "disconnected send");
                    Assert(events == 0, "Rejected send emitted transport activity.");
                }
            });

            Case(results, "rdk_exact_set_wire_and_query_final", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    Assert(await fixture.Read(stream) == "CHMODE= 1", "Set command needs exactly one space after tag.");
                    await Fixture.Write(stream, "CHMODE: +\r\n");
                    Assert(await fixture.Read(stream) == "ZPOS?", "Query wire changed.");
                    await Fixture.Write(stream, "ZPOS: 12.5\r\n");
                }))
                {
                    await f.Connect();
                    RdkReply reply = await f.Client.SendAsync("CHMODE=1", 2000, "takeover");
                    Assert(!reply.IsError && reply.Raw == "CHMODE: +", "Set final missing.");
                    reply = await f.Client.SendAsync("ZPOS?", 2000, "position");
                    Assert(reply.Value == "12.5" && reply.RequestId == "position", "Query association wrong.");
                    Assert(f.HasEvent("physical_completion = False"), "Replies must not assert physical completion.");
                    Assert(f.ServerFailure == null, "Server assertion failed.");
                }
            });

            Case(results, "rdk_split_utf8_and_crlf_preserves_opaque_query", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    byte[] data = Encoding.UTF8.GetBytes("GETERRCODE: 故障测试\r\n");
                    for (int i = 0; i < data.Length; i++)
                    {
                        await stream.WriteAsync(data, i, 1);
                        await Task.Delay(2);
                    }
                }))
                {
                    await f.Connect();
                    RdkReply reply = await f.Client.SendAsync("GETERRCODE?", 3000, "unicode");
                    Assert(reply.Raw == "GETERRCODE: 故障测试", "Incremental UTF-8 corrupted payload.");
                }
            });

            Case(results, "rdk_init_and_coalesced_temporary_final", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await Fixture.Write(stream, "INITNRML+\r\nINITNRML: +\r\n");
                    await fixture.Read(stream);
                    await Fixture.Write(stream, "MVZR+\r\nMVZR: +\r\n");
                }))
                {
                    await f.Connect();
                    await WaitFor(delegate { return f.HasEvent("initialization_final_notice"); }, "Missing startup notices");
                    Assert(!f.Client.Unknown, "INIT notices should not create a command or fail transport.");
                    RdkReply reply = await f.Client.SendAsync("MVZR=2", 2000, "move");
                    Assert(reply.Value == "+" && f.HasEvent("temporary_acceptance_reply"), "Temp/final coalescing lost.");
                    Assert(f.Commands.Count == 1 && f.Commands[0] == "MVZR= 2", "Client sent unexpected INIT or command.");
                }
            });

            Case(results, "rdk_temporary_acceptance_does_not_finish", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    await Fixture.Write(stream, "MVZR+\r\n");
                    await Task.Delay(300);
                    await Fixture.Write(stream, "MVZR: +\r\n");
                }))
                {
                    await f.Connect();
                    Task<RdkReply> task = f.Client.SendAsync("MVZR=2", 2000, "temporary");
                    await WaitFor(delegate { return f.HasEvent("temporary_acceptance_reply"); }, "Missing acceptance");
                    Assert(!task.IsCompleted && f.Client.Busy, "Acceptance was treated as final.");
                    await task;
                }
            });

            Case(results, "rdk_set_and_query_error_are_not_success", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    await Fixture.Write(stream, "CHMODE! 1\r\nCHMODE: ! 1\r\n");
                    await fixture.Read(stream);
                    await Fixture.Write(stream, "ZPOS: ! 3\r\n");
                }))
                {
                    await f.Connect();
                    RdkReply a = await f.Client.SendAsync("CHMODE=1", 2000, "set-error");
                    RdkReply b = await f.Client.SendAsync("ZPOS?", 2000, "query-error");
                    Assert(a.IsError && b.IsError && b.Raw == "ZPOS: ! 3", "Error payload reinterpreted as successful data.");
                    Assert(f.HasEvent("temporary_error_reply"), "Temporary error evidence missing.");
                }
            });

            Case(results, "rdk_busy_rejects_without_queue_or_send", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    await Task.Delay(300);
                    await Fixture.Write(stream, "MVZR: +\r\n");
                    await Task.Delay(100);
                }))
                {
                    await f.Connect();
                    Task<RdkReply> first = f.Client.SendAsync("MVZR=2", 2000, "first");
                    await ExpectFailure(f.Client.SendAsync("ZPOS?", 2000, "second"), "busy send");
                    await first;
                    Assert(f.Commands.Count == 1 && !f.HasEvent("|second|"), "Rejected request was queued or recorded as sent.");
                }
            });

            Case(results, "rdk_timeout_unknown_no_retry_late_reply_not_reused", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    await Task.Delay(450);
                    try { await Fixture.Write(stream, "MVZR: +\r\n"); } catch (IOException) { }
                    try { await fixture.Read(stream); } catch (IOException) { }
                }))
                {
                    await f.Connect();
                    await ExpectFailure(f.Client.SendAsync("MVZR=2", 100, "timeout"), "timeout");
                    Assert(f.Client.Unknown && !f.Client.Connected, "Timed-out connection not quarantined.");
                    await ExpectFailure(f.Client.SendAsync("MVZR=2", 500, "no-retry"), "after timeout");
                    await Task.Delay(500);
                    Assert(f.Commands.Count == 1, "Relative move retried.");
                    Assert(!f.HasEvent("final_reply|"), "Late reply satisfied an expired or next request.");
                }
            });

            Case(results, "rdk_eof_fails_pending_and_marks_unknown", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    stream.Close();
                }))
                {
                    await f.Connect();
                    await ExpectFailure(f.Client.SendAsync("SHOTSTS?", 2000, "eof"), "EOF");
                    Assert(f.Client.Unknown && !f.Client.Connected, "EOF interpreted as stop/success.");
                }
            });

            Case(results, "rdk_unmatched_reply_is_not_buffered_for_future", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    await Fixture.Write(stream, "ZPOS: 11\r\n");
                }))
                {
                    await f.Connect();
                    await ExpectFailure(f.Client.SendAsync("STGPOS?", 2000, "xy"), "unmatched");
                    await ExpectFailure(f.Client.SendAsync("ZPOS?", 2000, "z"), "reuse stray reply");
                    Assert(f.Client.Unknown && f.HasEvent("unmatched_or_duplicate_response"), "Unmatched data lost.");
                }
            });

            Case(results, "rdk_duplicate_final_quarantines_connection", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    await Fixture.Write(stream, "ZPOS: 3\r\nZPOS: 3\r\n");
                }))
                {
                    await f.Connect();
                    // First final can return, but duplicate must poison further control.
                    try { await f.Client.SendAsync("ZPOS?", 2000, "duplicate"); } catch (IOException) { }
                    await WaitFor(delegate { return f.Client.Unknown; }, "Duplicate final not quarantined");
                    await ExpectFailure(f.Client.SendAsync("ZPOS?", 1000, "next"), "duplicate next");
                }
            });

            Case(results, "rdk_invalid_utf8_and_bad_line_boundary_rejected", async delegate
            {
                foreach (byte[] payload in new byte[][] { new byte[] { 0xff, 13, 10 }, Encoding.UTF8.GetBytes("ZPOS: 1\n") })
                {
                    byte[] current = payload;
                    using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                    {
                        await fixture.Read(stream);
                        await stream.WriteAsync(current, 0, current.Length);
                    }))
                    {
                        await f.Connect();
                        await ExpectFailure(f.Client.SendAsync("ZPOS?", 2000, "bad-bytes"), "bad framing");
                        Assert(f.Client.Unknown, "Malformed response did not halt control.");
                    }
                }
            });

            Case(results, "rdk_white_list_blocks_injection_and_fractional_moves", async delegate
            {
                using (RdkClient client = new RdkClient(delegate(string s, string k, object d, string r, string c) { }))
                {
                    string[] invalid = { "GETSTS?", "INITNRML=", "SHUTDOWN=1", "ACQMODE=0", "ENDSHOT=1", "MVZR=1.5", "MVZR=1,5", "MVZR=2147483648", "MVSTG=1,2,3", "MVSTG=1.1,2,0", "CHMODE=1\r\nMVZR=20", "CHMODE=  1", "SHOT=1,C:\\two words\\x", "SHOT=1,C:\\a,b\\x", "SHOT=0,C:\\a\\x", "SHOT=1,\\\\host\\x", "SHOT=1,C:\\a\\..\\x", "ZPOS? ", "mvzr=2", "CHMODE=1\t" };
                    foreach (string command in invalid)
                    {
                        bool rejected = false;
                        try { RdkClient.ValidateAndFormat(command); } catch (ArgumentException) { rejected = true; }
                        Assert(rejected, "Whitelist accepted: " + command);
                    }
                    Assert(RdkClient.ValidateAndFormat("MVSTG=-2,3,0") == "MVSTG= -2,3,0", "Valid XY format rejected.");
                    Assert(RdkClient.ValidateAndFormat("SHOT=1,C:\\DSX\\sample_0001") == "SHOT= 1,C:\\DSX\\sample_0001", "Valid shot path rejected.");
                    await Task.FromResult(0);
                }
            });

            Case(results, "rdk_sensitive_unexpected_payload_is_redacted", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    await Fixture.Write(stream, "unexpected /PASS:TEST_SECRET password=TEST_SECRET\r\n");
                }))
                {
                    await f.Connect();
                    await ExpectFailure(f.Client.SendAsync("ZPOS?", 2000, "redact"), "unexpected secret");
                    Assert(!f.HasEvent("TEST_SECRET") && f.HasEvent("REDACTED"), "Unexpected payload leaked credential fixture.");
                }
            });

            Case(results, "rdk_explicit_disconnect_does_not_emit_stop_or_release", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream); // will return EOF; no control should be written
                }))
                {
                    await f.Connect();
                    f.Client.Disconnect();
                    await f.Server;
                    Assert(f.Commands.Count == 0 && f.Client.Unknown, "Disconnect sent an implicit command.");
                    Assert(f.HasEvent("device_stopped = False"), "Disconnect falsely reported stop.");
                }
            });

            Case(results, "rdk_initialization_error_blocks_control", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await Fixture.Write(stream, "INITNRML: ! 3\r\n");
                }))
                {
                    await f.Connect();
                    await WaitFor(delegate { return f.Client.Unknown; }, "Startup error not quarantined");
                    Assert(f.HasEvent("initialization_final_notice"), "Initialization error lost.");
                    await ExpectFailure(f.Client.SendAsync("CHMODE=1", 500, "blocked-init"), "failed initialization");
                    Assert(f.Commands.Count == 0, "Control sent after failed initialization.");
                }
            });

            Case(results, "rdk_disconnect_interrupts_pending_without_stop_claim", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                    await Task.Delay(300);
                }))
                {
                    await f.Connect();
                    Task<RdkReply> active = f.Client.SendAsync("MVZR=2", 2000, "interrupted");
                    await WaitFor(delegate { lock (f.Commands) return f.Commands.Count == 1; }, "Missing in-flight command");
                    f.Client.Disconnect();
                    await ExpectFailure(active, "disconnect while pending");
                    Assert(f.Client.Unknown && !f.Client.Busy, "Pending state remained after disconnect.");
                    Assert(f.Commands.Count == 1, "Disconnect sent implicit stop/release.");
                }
            });

            Case(results, "rdk_evidence_write_failure_prevents_command_send", async delegate
            {
                using (Fixture f = new Fixture(async delegate(Fixture fixture, NetworkStream stream)
                {
                    await fixture.Read(stream);
                }))
                using (RdkClient client = new RdkClient(delegate(string s, string k, object d, string r, string c)
                {
                    if (k == "command_requested") throw new IOException("Simulated disk-full evidence failure.");
                }))
                {
                    await client.ConnectAsync("127.0.0.1", f.Port, 2000);
                    await ExpectFailure(client.SendAsync("MVZR=2", 2000, "disk-full"), "evidence failure");
                    await f.Server;
                    Assert(f.Commands.Count == 0 && client.Unknown, "Command transmitted without evidence or failure was ignored.");
                }
            });
            return results;
        }
    }
}
