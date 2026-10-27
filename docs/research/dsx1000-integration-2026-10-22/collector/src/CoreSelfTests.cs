using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;

namespace DsxCollector
{
    // These tests use temporary local files only. They neither discover nor connect to RDK.
    public static class CoreSelfTests
    {
        private static void Check(bool value, string detail) { if (!value) throw new InvalidOperationException(detail); }
        private static void Case(List<object> results, string name, Action action)
        {
            try { action(); results.Add(Util.Obj("name", name, "passed", true, "detail", "Passed using local synthetic data; no DSX hardware.")); }
            catch (Exception ex) { results.Add(Util.Obj("name", name, "passed", false, "detail", Util.Redact(ex.ToString()))); }
        }
        private static List<Dictionary<string, object>> Events(string directory)
        {
            JavaScriptSerializer json = new JavaScriptSerializer();
            return ReadSharedLines(Path.Combine(directory, "events.jsonl")).Where(l => l.Length > 0).Select(l => json.Deserialize<Dictionary<string, object>>(l)).ToList();
        }
        private static string ReadSharedText(string path)
        {
            using (FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            using (StreamReader reader = new StreamReader(stream, Encoding.UTF8, true)) return reader.ReadToEnd();
        }
        private static string[] ReadSharedLines(string path)
        {
            List<string> lines = new List<string>();
            using (FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            using (StreamReader reader = new StreamReader(stream, Encoding.UTF8, true))
            { string line; while ((line = reader.ReadLine()) != null) lines.Add(line); }
            return lines.ToArray();
        }
        private static bool Has(List<Dictionary<string, object>> events, string kind) { return events.Any(e => Convert.ToString(e["kind"]) == kind); }
        public static List<object> Run(string root)
        {
            List<object> results = new List<object>();
            string testRoot = Path.Combine(Path.GetFullPath(root), "core-selftest-中文-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(testRoot);
            Case(results, "json_unicode_and_credential_redaction", delegate
            {
                string json = Util.Json(Util.Obj("text", "中文 /PASS:\"credential one\" /ID:operator", "password", "credential two", "nested", new[] { "token=credential three" }));
                Check(!json.Contains("credential"), "A credential leaked through JSON serialization.");
                new JavaScriptSerializer().DeserializeObject(json);
                string file = Path.Combine(testRoot, "中文.json"); Util.WriteJson(file, Util.Obj("value", "初值")); Util.WriteJson(file, Util.Obj("value", "替换"));
                Check(ReadSharedText(file).Contains("替换"), "Atomic replacement failed.");
                Check(!Directory.GetFiles(testRoot, "*.writing-*").Any(), "A temporary write file remained.");
            });
            Case(results, "quoted_json_xml_and_escaped_credential_redaction", delegate
            {
                string[] examples = {
                    "{\"password\":\"SENTINEL-A\"}", "{'pass':'SENTINEL-A'}", "{\"access_token\":\"SENTINEL-A\"}",
                    "{\\\"token\\\":\\\"SENTINEL-A\\\"}", "<Password>SENTINEL-A</Password>",
                    "<ns:Password>\nSENTINEL-A\n</ns:Password>", "<account password=\"SENTINEL-A\" />" };
                foreach (string example in examples) Check(!Util.Redact(example).Contains("SENTINEL-A"), "Quoted/structured credential leaked: " + Util.Redact(example));
                Check(!Util.Json(Util.Obj("refresh_token", "SENTINEL-A")).Contains("SENTINEL-A"), "Structured refresh token leaked.");
            });
            Case(results, "session_concurrent_events_csv_and_unique_ids", delegate
            {
                using (EvidenceSession session = new EvidenceSession(testRoot))
                {
                    session.NewRun("实验一", Util.Obj("target_xyz", new[] { 1, 2, 3 }, "reported_xyz", null));
                    Thread[] writers = Enumerable.Range(0, 4).Select(index => new Thread(delegate() { for (int i = 0; i < 25; i++) session.Add("test", "thread_event", Util.Obj("thread", index, "value", "逗号,和\"引号"), "r" + index, null); })).ToArray();
                    foreach (Thread writer in writers) writer.Start(); foreach (Thread writer in writers) writer.Join();
                    List<Dictionary<string, object>> events = Events(session.DirectoryPath);
                    Check(events.Count(e => Convert.ToString(e["kind"]) == "thread_event") == 100, "Concurrent event loss.");
                    long previous = 0; foreach (Dictionary<string, object> item in events) { long current = Convert.ToInt64(item["sequence"]); Check(current == previous + 1, "Non-contiguous event order."); previous = current; }
                    Check(ReadSharedLines(Path.Combine(session.DirectoryPath, "timeline.csv")).Length == events.Count + 1, "CSV/JSONL event count differs.");
                }
            });
            Case(results, "watch_slow_write_rewrite_rename_and_old_file", delegate
            {
                string images = Path.Combine(testRoot, "慢写图像"); Directory.CreateDirectory(images);
                string old = Path.Combine(images, "旧图.tif"); File.WriteAllText(old, "old");
                using (EvidenceSession session = new EvidenceSession(testRoot))
                using (ObservationRecorder recorder = new ObservationRecorder(session, images, null))
                {
                    recorder.Start(); string image = Path.Combine(images, "new.tif");
                    using (FileStream stream = new FileStream(image, FileMode.CreateNew, FileAccess.Write, FileShare.ReadWrite))
                    {
                        stream.WriteByte(1); stream.Flush(); Thread.Sleep(700); stream.WriteByte(2); stream.Flush(); Thread.Sleep(700);
                        stream.Position = 0; stream.WriteByte(3); stream.Flush();
                    }
                    Thread.Sleep(2600); File.Move(image, Path.Combine(images, "renamed.tif")); Thread.Sleep(700); recorder.Stop();
                    List<Dictionary<string, object>> events = Events(session.DirectoryPath);
                    Check(Has(events, "baseline_file"), "Existing image was not labelled baseline.");
                    Check(Has(events, "file_created_observed") && Has(events, "file_changed_observed"), "Slow write/change observations missing.");
                    Check(Has(events, "stable_candidate") && !Has(events, "capture_complete"), "Stable file was incorrectly treated as capture completion.");
                    Check(Has(events, "rename_observed") || Has(events, "file_missing_observed"), "Rename was not observable.");
                    Check(recorder.ObservedFiles.Contains(old), "Old image no longer available with its baseline evidence.");
                }
            });
            Case(results, "log_append_split_credential_rotation_and_partial_line", delegate
            {
                string logs = Path.Combine(testRoot, "日志"); Directory.CreateDirectory(logs);
                string log = Path.Combine(logs, "vendor.log"); File.WriteAllText(log, "old\n/PASS:baseline", new UTF8Encoding(false));
                using (EvidenceSession session = new EvidenceSession(testRoot))
                using (ObservationRecorder recorder = new ObservationRecorder(session, null, logs))
                {
                    recorder.Start(); File.AppendAllText(log, "_secret\n正常行\n/PASS:", Encoding.UTF8); Thread.Sleep(650);
                    File.AppendAllText(log, "split_secret\n", Encoding.UTF8); Thread.Sleep(650);
                    File.WriteAllText(log, "rotation\n", new UTF8Encoding(false)); Thread.Sleep(650);
                    File.AppendAllText(log, "password=final_secret", Encoding.UTF8); Thread.Sleep(650); recorder.Stop(); recorder.CopySamples(new[] { log });
                    string content = ReadSharedText(Path.Combine(session.DirectoryPath, "events.jsonl"));
                    Check(!content.Contains("_secret"), "Credential fragment leaked in evidence.");
                    List<Dictionary<string, object>> events = Events(session.DirectoryPath);
                    Check(Has(events, "log_reset_observed"), "Log truncate/rotation not recorded.");
                    Check(Has(events, "baseline_partial_line_skipped") && Has(events, "log_final_partial"), "Partial line boundary handling missing.");
                    foreach (string copy in Directory.GetFiles(Path.Combine(session.DirectoryPath, "samples"))) Check(!ReadSharedText(copy).Contains("final_secret"), "Raw credential copied with log.");
                }
            });
            Case(results, "utf16_log_and_shared_read_copy", delegate
            {
                string logs = Path.Combine(testRoot, "utf16"); Directory.CreateDirectory(logs);
                string log = Path.Combine(logs, "unicode.log"); File.WriteAllText(log, "原有\n", Encoding.Unicode);
                using (EvidenceSession session = new EvidenceSession(testRoot))
                using (ObservationRecorder recorder = new ObservationRecorder(session, null, logs))
                {
                    recorder.Start(); File.AppendAllText(log, "新增汉字\n", Encoding.Unicode); Thread.Sleep(650); recorder.Stop(); recorder.CopySamples(new[] { log });
                    Check(ReadSharedText(Path.Combine(session.DirectoryPath, "events.jsonl")).Contains("新增汉字"), "UTF-16 appended line missing.");
                    Check(Has(Events(session.DirectoryPath), "copy_verified"), "Sanitized stable log copy not verified.");
                }
            });
            Case(results, "xml_multiline_and_json_log_copy_are_redacted", delegate
            {
                string logs = Path.Combine(testRoot, "structured-logs"); Directory.CreateDirectory(logs);
                using (EvidenceSession session = new EvidenceSession(testRoot))
                using (ObservationRecorder recorder = new ObservationRecorder(session, null, logs))
                {
                    recorder.Start();
                    string xml = Path.Combine(logs, "vendor.xml"), json = Path.Combine(logs, "vendor.json");
                    File.WriteAllText(xml, "<Password>\nSENTINEL-B\n</Password>\n<status>ok</status>\n", new UTF8Encoding(false));
                    File.WriteAllText(json, "{\"password\":\"SENTINEL-C\"}\n", new UTF8Encoding(false));
                    Thread.Sleep(750); recorder.Stop(); recorder.CopySamples(new[] { xml, json });
                    foreach (string file in Directory.GetFiles(session.DirectoryPath, "*", SearchOption.AllDirectories))
                        Check(!ReadSharedText(file).Contains("SENTINEL-"), "Structured log credential leaked into session.");
                    Check(Events(session.DirectoryPath).Count(e => Convert.ToString(e["kind"]) == "copy_verified") == 2, "Expected two sanitized log copies.");
                }
            });
            Case(results, "image_sample_whitelist_blocks_arbitrary_files", delegate
            {
                string images = Path.Combine(testRoot, "sample-whitelist"); Directory.CreateDirectory(images);
                string image = Path.Combine(images, "allowed.tif"), text = Path.Combine(images, "settings.txt"), binary = Path.Combine(images, "settings.bin");
                File.WriteAllText(image, "synthetic image bytes"); File.WriteAllText(text, "password=SENTINEL-D"); File.WriteAllText(binary, "SENTINEL-E");
                using (EvidenceSession session = new EvidenceSession(testRoot))
                using (ObservationRecorder recorder = new ObservationRecorder(session, images, null))
                {
                    recorder.Start(); recorder.Stop(); recorder.CopySamples(new[] { image, text, binary });
                    string[] copies = Directory.GetFiles(Path.Combine(session.DirectoryPath, "samples"));
                    Check(copies.Length == 1 && Path.GetExtension(copies[0]) == ".tif", "A non-image file was copied from the image directory.");
                    Check(Events(session.DirectoryPath).Count(e => Convert.ToString(e["kind"]) == "collection_error") == 2, "Blocked sample types need explicit errors.");
                    foreach (string file in Directory.GetFiles(session.DirectoryPath, "*", SearchOption.AllDirectories))
                        Check(!ReadSharedText(file).Contains("SENTINEL-"), "Blocked arbitrary-file contents leaked into session.");
                }
            });
            Case(results, "locked_sample_and_invalid_selection_are_reported", delegate
            {
                string images = Path.Combine(testRoot, "locked"); Directory.CreateDirectory(images); string image = Path.Combine(images, "locked.tif"); File.WriteAllText(image, "test");
                using (EvidenceSession session = new EvidenceSession(testRoot))
                using (ObservationRecorder recorder = new ObservationRecorder(session, images, null))
                {
                    recorder.Start(); recorder.Stop();
                    using (FileStream locked = new FileStream(image, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) recorder.CopySamples(new[] { image });
                    recorder.CopySamples(new[] { Path.Combine(testRoot, "not_observed.tif") });
                    Check(Events(session.DirectoryPath).Count(e => Convert.ToString(e["kind"]) == "collection_error") >= 2, "Copy errors were silently swallowed.");
                }
            });
            Case(results, "observer_rejects_evidence_directory_overlap", delegate
            {
                using (EvidenceSession session = new EvidenceSession(testRoot))
                {
                    bool rejected = false; try { using (ObservationRecorder recorder = new ObservationRecorder(session, testRoot, null)) { } } catch (ArgumentException) { rejected = true; }
                    Check(rejected, "Evidence output could observe itself recursively.");
                }
            });
            Case(results, "export_zip_manifest_hash_and_sealed_session", delegate
            {
                using (EvidenceSession session = new EvidenceSession(testRoot))
                {
                    session.Add("test", "sample", Util.Obj("value", 1)); string zip = session.ExportZip();
                    Check(File.Exists(zip) && ReadSharedText(zip + ".sha256").StartsWith(Util.Sha256(zip), StringComparison.Ordinal), "ZIP sidecar hash mismatch.");
                    string unpack = Path.Combine(testRoot, "unpacked-" + Guid.NewGuid().ToString("N")); ZipFile.ExtractToDirectory(zip, unpack);
                    Dictionary<string, object> manifest = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(ReadSharedText(Path.Combine(unpack, "manifest.json")));
                    foreach (object obj in (IEnumerable)manifest["files"])
                    {
                        IDictionary<string, object> item = (IDictionary<string, object>)obj;
                        Check(Util.Sha256(Path.Combine(unpack, Convert.ToString(item["path"]))) == Convert.ToString(item["sha256"]), "Manifest hash mismatch.");
                    }
                    bool rejected = false; try { session.Add("test", "after_close", null); } catch (InvalidOperationException) { rejected = true; }
                    Check(rejected, "Session accepted events after export.");
                }
            });
            Case(results, "partial_session_readonly_recovery_and_corruption", delegate
            {
                string directory;
                using (EvidenceSession session = new EvidenceSession(testRoot)) { directory = session.DirectoryPath; session.Add("test", "complete_before_crash", null); }
                File.AppendAllText(Path.Combine(directory, "events.jsonl"), "{\"sequence\":", Encoding.UTF8); File.WriteAllText(Path.Combine(directory, "summary.json"), "{broken");
                string before = Util.Sha256(Path.Combine(directory, "events.jsonl")); Dictionary<string, object> report = EvidenceSession.InspectRecovery(directory);
                Check(Convert.ToInt64(report["valid_event_lines"]) >= 2 && !Convert.ToBoolean(report["complete"]), "Partial recovery failed to distinguish valid and corrupt evidence.");
                Check(before == Util.Sha256(Path.Combine(directory, "events.jsonl")), "Recovery modified original evidence.");
            });
            Case(results, "seal_preserves_closed_partial_status", delegate
            {
                string directory;
                using (EvidenceSession session = new EvidenceSession(testRoot))
                { directory = session.DirectoryPath; session.SaveSummary("closed_partial"); }
                Dictionary<string, object> summary = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(ReadSharedText(Path.Combine(directory, "summary.json")));
                Check(Convert.ToString(summary["status"]) == "closed_partial", "Sealing overwrote the partial-session status.");
            });
            Case(results, "summary_evidence_coverage_counts_missing_and_partial", delegate
            {
                string directory;
                using (EvidenceSession session = new EvidenceSession(testRoot))
                {
                    directory = session.DirectoryPath;
                    Dictionary<string, object> initial = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(ReadSharedText(Path.Combine(directory, "summary.json")));
                    Check(((IEnumerable)initial["not_collected"]).Cast<object>().Count() == 6, "Initial summary must identify all six uncollected categories.");
                    session.NewRun("synthetic coverage", Util.Obj("source", "self_test"));
                    session.Add("inventory", "snapshot", Util.Obj("synthetic", true));
                    session.Add("filesystem_poll", "baseline_file", Util.Obj("path", "old.tif"));
                    session.Add("filesystem_poll", "file_created_observed", Util.Obj("path", "not-an-image.txt"));
                    session.SaveSummary("recording");
                    Dictionary<string, object> partial = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(ReadSharedText(Path.Combine(directory, "summary.json")));
                    Check(((IEnumerable)partial["not_collected"]).Cast<object>().Count() == 4, "Baseline/ordinary text must not count as new image evidence.");
                    session.Add("filesystem_poll", "file_created_observed", Util.Obj("path", "new.tif"));
                    session.Add("log_tail", "log_line", Util.Obj("text", "synthetic line"));
                    session.Add("sample_copy", "copy_verified", Util.Obj("synthetic", true));
                    session.Add("rdk", "command_requested", Util.Obj("synthetic", true, "sent", false));
                    session.AddError("synthetic_gap", new IOException("test gap remains independent of coverage"));
                    session.SaveSummary("closed_partial");
                }
                Dictionary<string, object> final = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(ReadSharedText(Path.Combine(directory, "summary.json")));
                IDictionary<string, object> counts = (IDictionary<string, object>)final["evidence_counts"], kinds = (IDictionary<string, object>)final["kind_counts"];
                Check(counts.Values.All(value => Convert.ToInt64(value) == 1), "Expected one recorded event per evidence category.");
                Check(((IEnumerable)final["not_collected"]).Cast<object>().Count() == 0, "Covered categories still marked missing.");
                Check(((IEnumerable)final["errors"]).Cast<object>().Count() == 1 && Convert.ToString(final["status"]) == "closed_partial", "Coverage must not clear errors or promote partial status.");
                Check(kinds.Values.Sum(value => Convert.ToInt64(value)) == Convert.ToInt64(final["event_count"]), "Successfully recorded kind counts differ from event count.");
            });
            Case(results, "passive_inventory_missing_installation_and_baseline", delegate
            {
                InventoryResult result = Inventory.Capture(Path.Combine(testRoot, "missing"), Path.Combine(testRoot, "missing-baseline.json"));
                Check(!result.Matched && Convert.ToString(result.Data["network_mode"]).Contains("no connect"), "Missing installation unexpectedly matched or passive contract absent.");
                Check(!Util.Json(result.Data).Contains("CommandLine"), "Unexpected command line collection.");
            });
            Case(results, "baseline_file_identity_match_and_mismatch", delegate
            {
                string install = Path.Combine(testRoot, "mock-install"); Directory.CreateDirectory(Path.Combine(install, "DSX-BSW-CTRL"));
                string file = Path.Combine(install, "DSX-BSW-CTRL", "mock.dll"); File.WriteAllText(file, "synthetic binary identity");
                string baseline = Path.Combine(testRoot, "baseline.json"); Util.WriteJson(baseline, Util.Obj("files", new[] { Util.Obj("path", "DSX-BSW-CTRL/mock.dll", "sha256", Util.Sha256(file)) }));
                Check(Inventory.Capture(install, baseline).Matched, "Synthetic baseline should match."); File.AppendAllText(file, "changed"); Check(!Inventory.Capture(install, baseline).Matched, "Changed identity should not match.");
            });
            Case(results, "unwritable_destination_error_keeps_existing_data", delegate
            {
                string occupied = Path.Combine(testRoot, "a-file-not-directory"); File.WriteAllText(occupied, "preserve");
                bool rejected = false; try { using (EvidenceSession session = new EvidenceSession(occupied)) { } } catch (IOException) { rejected = true; }
                Check(rejected && ReadSharedText(occupied) == "preserve", "Invalid destination must fail without replacing existing data.");
            });
            return results;
        }
    }
}
