using System;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace DsxCollector
{
    public sealed class RdkReply
    {
        public string Command, Raw, Value, RequestId;
        public bool IsError;
    }

    // This is a recorder of protocol replies, not a motion/capture completion oracle.
    // Wire evidence: OlymRdkData CRdkCommandManager.SearchCommandInfo (0x0600001f),
    // GetParameterString (0x0600001e), CRdkCommandInfo.CreateTempReplyCommandString
    // and CreateFinalReplyCommandString (0x06000010/11), selected-methods.json.
    // Set tags require a SPACE after '='; final replies have ': ', temporary replies do not.
    public sealed class RdkClient : IDisposable
    {
        private sealed class Pending
        {
            public string Name, Wire, RequestId, CaptureId;
            public bool IsQuery;
            public readonly TaskCompletionSource<RdkReply> Completion =
                new TaskCompletionSource<RdkReply>((TaskCreationOptions)64); // RunContinuationsAsynchronously (.NET 4.8).
        }

        private readonly object gate = new object();
        private readonly Action<string, string, object, string, string> record;
        private TcpClient socket;
        private Pending pending;
        private bool connected, unknown, connecting, disposed;
        private int generation;
        private static readonly Regex FinalLine = new Regex(@"\A([A-Z][A-Z0-9]*): (.+)\z", RegexOptions.CultureInvariant);
        private static readonly Regex TempLine = new Regex(@"\A([A-Z][A-Z0-9]*)(\+|! [0-9]+)\z", RegexOptions.CultureInvariant);
        public event Action StateChanged;

        public RdkClient(Action<string, string, object, string, string> record)
        {
            if (record == null) throw new ArgumentNullException("record");
            this.record = record;
        }

        public bool Connected { get { lock (gate) return connected; } }
        public bool Unknown { get { lock (gate) return unknown; } }
        public bool Busy { get { lock (gate) return connecting || pending != null; } }

        private void Notify()
        {
            Action handler = StateChanged;
            if (handler != null) handler();
        }

        private void Log(string kind, object data, Pending request)
        {
            record("rdk", kind, data, request == null ? null : request.RequestId,
                   request == null ? null : request.CaptureId);
        }

        private static string Scrub(string text)
        {
            return Regex.Replace(text ?? "", @"(?i)(/PASS:|password\s*[=:]\s*)(?:""[^""]*""|[^\s,;]+)", "$1[REDACTED]");
        }

        private static void TimeoutArgument(int timeoutMs)
        {
            if (timeoutMs < 1 || timeoutMs > 600000) throw new ArgumentOutOfRangeException("timeoutMs");
        }

        public async Task ConnectAsync(string host, int port, int timeoutMs)
        {
            TimeoutArgument(timeoutMs);
            if (String.IsNullOrWhiteSpace(host) || host.IndexOfAny(new char[] { '\r', '\n', '/', '@' }) >= 0)
                throw new ArgumentException("请输入原厂监听表中确认的主机地址。", "host");
            if (port < 1 || port > 65535) throw new ArgumentOutOfRangeException("port");
            IPAddress literalAddress;
            TcpClient client = IPAddress.TryParse(host, out literalAddress)
                ? new TcpClient(literalAddress.AddressFamily) : new TcpClient();
            int version;
            lock (gate)
            {
                if (disposed) throw new ObjectDisposedException("RdkClient");
                if (connected || connecting || pending != null) throw new InvalidOperationException("已有连接或请求；禁止并行连接。");
                connecting = true;
                socket = client;
                version = ++generation;
            }
            Notify();
            try
            {
                Log("connect_requested", new { host = host, port = port, first_connection_may_initialize_vendor = true }, null);
                Task attempt = client.ConnectAsync(host, port);
                if (await Task.WhenAny(attempt, Task.Delay(timeoutMs)).ConfigureAwait(false) != attempt)
                {
                    Observe(attempt);
                    throw new TimeoutException("连接超时；状态未知，未重试。");
                }
                await attempt.ConfigureAwait(false);
                client.NoDelay = true;
                lock (gate)
                {
                    if (generation != version || disposed) throw new IOException("连接已被手动关闭。");
                    connected = true;
                    connecting = false;
                    unknown = false;
                }
                Log("connected_transport_only", new { host = host, port = port, initialization_confirmed = false }, null);
                Observe(ReadLoop(client, version));
                Notify();
            }
            catch (Exception error)
            {
                Fail(version, error, "connect_failure");
                client.Close();
                throw;
            }
        }

        // Input can use either canonical wire form ('CHMODE= 1') or compact UI form
        // ('CHMODE=1'); only the former is ever transmitted. No path quoting is invented.
        internal static string ValidateAndFormat(string command)
        {
            if (String.IsNullOrEmpty(command) || command.Length > 2048)
                throw new ArgumentException("命令为空或过长。", "command");
            foreach (char c in command)
                if (Char.IsControl(c)) throw new ArgumentException("命令禁止换行和控制字符。", "command");
            if (command == "STGPOS?" || command == "ZPOS?" || command == "GETERRCODE?" || command == "SHOTSTS?")
                return command;
            int equals = command.IndexOf('=');
            if (equals < 1) throw new ArgumentException("命令不在受控验证白名单中。", "command");
            string name = command.Substring(0, equals);
            string args = command.Substring(equals + 1);
            if (args.StartsWith(" ", StringComparison.Ordinal)) args = args.Substring(1);
            foreach (char c in args)
                if (Char.IsWhiteSpace(c)) throw new ArgumentException("原厂参数不支持空格；路径请使用无空格独立目录。", "command");
            bool valid = false;
            if (name == "CHMODE") valid = args == "0" || args == "1";
            else if (name == "ACQMODE") valid = args == "1";
            else if (name == "ENDSHOT") valid = args == "0";
            else if (name == "MVZR") valid = IsInt32(args);
            else if (name == "MVSTG")
            {
                string[] values = args.Split(',');
                valid = values.Length == 3 && IsInt32(values[0]) && IsInt32(values[1]) &&
                    (values[2] == "0" || values[2] == "1" || values[2] == "2");
            }
            else if (name == "SHOT" && args.StartsWith("1,", StringComparison.Ordinal))
            {
                string path = args.Substring(2);
                // Deliberately require a local, absolute Windows path, not UNC/relative.
                // Unique basename and selected directory ownership are enforced by the UI.
                valid = Regex.IsMatch(path, @"\A[A-Za-z]:\\[^,:*?""<>|]+\z") &&
                    !path.EndsWith("\\", StringComparison.Ordinal) && path.Length < 240 &&
                    path.IndexOf('/') < 0 && path.IndexOf('=') < 0;
                foreach (string component in path.Substring(Math.Min(3, path.Length)).Split('\\'))
                    if (component == "." || component == ".." || component.Length == 0 || component.EndsWith(".", StringComparison.Ordinal)) valid = false;
            }
            if (!valid) throw new ArgumentException("命令参数不在已验证范围；Z/XY 必须为 Int32 整数微米，不做隐式取整。", "command");
            return name + "= " + args;
        }

        private static bool IsInt32(string value)
        {
            int parsed;
            return Regex.IsMatch(value, @"\A-?[0-9]+\z") &&
                Int32.TryParse(value, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out parsed);
        }

        public async Task<RdkReply> SendAsync(string command, int timeoutMs, string requestId, string captureId = null)
        {
            TimeoutArgument(timeoutMs);
            string wire = ValidateAndFormat(command);
            if (String.IsNullOrWhiteSpace(requestId) || requestId.Length > 200 || Regex.IsMatch(requestId, @"[\r\n]"))
                throw new ArgumentException("请求身份必须非空且不含换行。", "requestId");
            Pending request = new Pending {
                Name = wire.Substring(0, wire.IndexOfAny(new char[] { '=', '?' })),
                Wire = wire, RequestId = requestId, CaptureId = captureId,
                IsQuery = wire.EndsWith("?", StringComparison.Ordinal)
            };
            TcpClient client;
            int version;
            lock (gate)
            {
                if (disposed) throw new ObjectDisposedException("RdkClient");
                if (!connected || unknown) throw new InvalidOperationException("连接未建立或设备状态未知；禁止发送指令。");
                if (pending != null) throw new InvalidOperationException("上一请求仍在进行；本工具不排队或并发发送。");
                pending = request;
                client = socket;
                version = generation;
            }
            Notify();
            try
            {
                // Record durably BEFORE sending a command. Evidence failure prevents transmission.
                Log("command_requested", new { command = wire, meaning = "request_only_not_physical_completion" }, request);
                Task<RdkReply> operation = WriteAndAwait(client, request);
                if (await Task.WhenAny(operation, Task.Delay(timeoutMs)).ConfigureAwait(false) != operation)
                {
                    Observe(operation);
                    throw new TimeoutException("请求超时；原厂任务可能仍在进行，禁止自动重发。");
                }
                return await operation.ConfigureAwait(false);
            }
            catch (Exception error)
            {
                Fail(version, error, "request_failure");
                throw;
            }
            finally
            {
                lock (gate) if (Object.ReferenceEquals(pending, request)) pending = null;
                Notify();
            }
        }

        private static async Task<RdkReply> WriteAndAwait(TcpClient client, Pending request)
        {
            byte[] bytes = new UTF8Encoding(false, true).GetBytes(request.Wire + "\r\n");
            await client.GetStream().WriteAsync(bytes, 0, bytes.Length).ConfigureAwait(false);
            return await request.Completion.Task.ConfigureAwait(false);
        }

        private async Task ReadLoop(TcpClient client, int version)
        {
            byte[] bytes = new byte[4096];
            char[] chars = new char[4096];
            Decoder decoder = new UTF8Encoding(false, true).GetDecoder();
            StringBuilder line = new StringBuilder();
            try
            {
                NetworkStream stream = client.GetStream();
                while (true)
                {
                    int count = await stream.ReadAsync(bytes, 0, bytes.Length).ConfigureAwait(false);
                    if (count == 0) throw new EndOfStreamException("原厂连接关闭；不代表设备停止。");
                    int used = decoder.GetChars(bytes, 0, count, chars, 0, false);
                    // Keep a coalesced receive batch atomic against the next SendAsync:
                    // an extra final in this same batch must never match a new request.
                    lock (gate)
                    for (int i = 0; i < used; i++)
                    {
                        char ch = chars[i];
                        if (ch == '\n')
                        {
                            if (line.Length == 0 || line[line.Length - 1] != '\r') throw new InvalidDataException("响应缺少 CRLF 边界。");
                            line.Length--;
                            HandleLine(line.ToString(), version);
                            line.Clear();
                        }
                        else
                        {
                            if (line.Length > 0 && line[line.Length - 1] == '\r') throw new InvalidDataException("响应中出现孤立 CR。");
                            if (Char.IsControl(ch) && ch != '\r') throw new InvalidDataException("响应含控制字符。");
                            line.Append(ch);
                            if (line.Length > 16384) throw new InvalidDataException("响应超过长度上限。");
                        }
                    }
                }
            }
            catch (Exception error) { Fail(version, error, "receive_failure"); }
        }

        private void HandleLine(string raw, int version)
        {
            Match final = FinalLine.Match(raw);
            Match temp = final.Success ? Match.Empty : TempLine.Match(raw);
            string name = final.Success ? final.Groups[1].Value : temp.Groups[1].Value;
            string value = final.Success ? final.Groups[2].Value : temp.Groups[2].Value;
            Pending request;
            lock (gate)
            {
                if (generation != version || !connected || unknown) return;
                request = pending;
            }
            if (!final.Success && !temp.Success)
            {
                Log("unparseable_response", new { raw = Scrub(raw) }, request);
                throw new InvalidDataException("响应格式不符合已证实协议。");
            }
            bool error = Regex.IsMatch(value, @"\A! [0-9]+\z");
            if (name == "INITNRML")
            {
                if (value != "+" && !error) throw new InvalidDataException("初始化通知格式无效。");
                Log(final.Success ? "initialization_final_notice" : "initialization_temporary_notice",
                    new { raw = Scrub(raw), is_error = error, transport_notification_only = true }, null);
                if (error) throw new InvalidDataException("原厂初始化报告错误；请核对原厂状态。");
                return;
            }
            if (request == null || request.Name != name || request.Completion.Task.IsCompleted)
            {
                Log("unmatched_or_duplicate_response", new { raw = Scrub(raw), cannot_attribute_to_next_request = true }, request);
                throw new InvalidDataException("未匹配或重复应答；无法排除其他客户端或迟到回复。");
            }
            if (temp.Success)
            {
                Log(error ? "temporary_error_reply" : "temporary_acceptance_reply",
                    new { raw = Scrub(raw), final_reply = false, physical_completion = false }, request);
                return;
            }
            if ((!request.IsQuery && value != "+" && !error) || (value.StartsWith("!", StringComparison.Ordinal) && !error))
                throw new InvalidDataException("最终应答格式无效。");
            RdkReply reply = new RdkReply {
                Command = request.Wire, Raw = Scrub(raw), Value = Scrub(value), RequestId = request.RequestId, IsError = error
            };
            Log("final_reply", new { raw = reply.Raw, value = reply.Value, is_error = error,
                physical_completion = false, file_completion = false }, request);
            request.Completion.TrySetResult(reply);
        }

        private void Fail(int version, Exception error, string kind)
        {
            TcpClient client;
            Pending request;
            lock (gate)
            {
                if (version != generation) return;
                ++generation;
                connected = false;
                connecting = false;
                unknown = true;
                client = socket;
                socket = null;
                request = pending;
            }
            if (client != null) client.Close();
            if (request != null) request.Completion.TrySetException(error);
            try { Log(kind, new { error_type = error.GetType().Name, state = "unknown", device_stopped = false, retry = false }, request); }
            finally { Notify(); }
        }

        public void Disconnect()
        {
            int version;
            lock (gate) version = generation;
            Fail(version, new IOException("操作者关闭连接；设备状态未知，未发送停止或释放指令。"), "operator_disconnect");
        }

        private static void Observe(Task task)
        {
            task.ContinueWith(delegate(Task faulted) { var ignored = faulted.Exception; },
                TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously);
        }

        public void Dispose()
        {
            lock (gate) { if (disposed) return; disposed = true; }
            Disconnect();
        }
    }
}
