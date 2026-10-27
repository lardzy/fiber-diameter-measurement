// C# 5 / .NET Framework 4.8. Passive evidence collection; never opens a network connection.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.NetworkInformation;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using Microsoft.Win32;

namespace DsxCollector
{
    public static class Util
    {
        private static JavaScriptSerializer Serializer() { return new JavaScriptSerializer { MaxJsonLength = Int32.MaxValue, RecursionLimit = 100 }; }
        public static string Json(object value)
        {
            JavaScriptSerializer serializer = Serializer();
            return serializer.Serialize(Sanitize(serializer.DeserializeObject(serializer.Serialize(value))));
        }
        private static object Sanitize(object value)
        {
            if (value is string) return Redact((string)value);
            IDictionary<string, object> map = value as IDictionary<string, object>;
            if (map != null)
            {
                Dictionary<string, object> result = new Dictionary<string, object>();
                foreach (KeyValuePair<string, object> item in map)
                    result[Redact(item.Key)] = Regex.IsMatch(item.Key, "^(password|passwd|pass|token|access_token|refresh_token|authorization|secret)$", RegexOptions.IgnoreCase)
                        ? (object)"[REDACTED]" : Sanitize(item.Value);
                return result;
            }
            IEnumerable sequence = value as IEnumerable;
            if (sequence != null) { List<object> result = new List<object>(); foreach (object item in sequence) result.Add(Sanitize(item)); return result; }
            return value;
        }
        public static string Redact(string text)
        {
            if (text == null) return null;
            // Redact the whole remaining command segment for /PASS: because quoted Windows arguments
            // may contain spaces, nested escapes or malformed quotes. Loss of log detail is explicit.
            text = Regex.Replace(text, @"(?im)(/PASS\s*:)[^\r\n]*", "$1[REDACTED_TO_EOL]");
            // Log text can contain quoted JSON keys, INI attributes or escaped JSON. Redact
            // through EOL rather than attempting to recover the exact value's quoting rules.
            text = Regex.Replace(text, @"(?im)((?:\\?[""'])?(?:password|passwd|pass|secret|access_token|refresh_token|token|authorization)(?:\\?[""'])?\s*[:=]\s*)[^\r\n]*", "$1[REDACTED_TO_EOL]");
            // XML element content may span lines. In a single tailed line an opening tag
            // without its closing tag is suppressed; following lines are guarded by Tail.
            text = Regex.Replace(text, @"(?is)(<(?:[A-Za-z_][\w.-]*:)?(?:password|passwd|pass|secret|access_token|refresh_token|token|authorization)\b[^>]*>).*?(</(?:[A-Za-z_][\w.-]*:)?(?:password|passwd|pass|secret|access_token|refresh_token|token|authorization)\s*>)", "$1[REDACTED]$2");
            text = Regex.Replace(text, @"(?im)(<(?:[A-Za-z_][\w.-]*:)?(?:password|passwd|pass|secret|access_token|refresh_token|token|authorization)\b[^>]*>)[^\r\n]*", "$1[REDACTED_TO_EOL]");
            return text;
        }
        internal static string RedactLogLine(string text, ref bool sensitiveXml)
        {
            const string tag = @"(?:[A-Za-z_][\w.-]*:)?(?:password|passwd|pass|secret|access_token|refresh_token|token|authorization)";
            bool opens = Regex.IsMatch(text, "<" + tag + @"\b[^>]*>", RegexOptions.IgnoreCase);
            bool closes = Regex.IsMatch(text, "</" + tag + @"\s*>", RegexOptions.IgnoreCase);
            if (sensitiveXml)
            {
                if (closes) sensitiveXml = false;
                return "[REDACTED_XML_CONTINUATION]";
            }
            if (opens && !closes) sensitiveXml = true;
            return Redact(text);
        }
        public static void WriteJson(string path, object value)
        {
            string tmp = path + ".writing-" + Guid.NewGuid().ToString("N");
            try
            {
                using (FileStream stream = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
                using (StreamWriter writer = new StreamWriter(stream, new UTF8Encoding(false)))
                { writer.Write(Json(value)); writer.WriteLine(); writer.Flush(); stream.Flush(true); }
                if (File.Exists(path)) File.Replace(tmp, path, null); else File.Move(tmp, path);
            }
            finally { if (File.Exists(tmp)) File.Delete(tmp); }
        }
        public static string Sha256(string path)
        {
            using (FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            using (SHA256 sha = SHA256.Create()) return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
        internal static string Csv(object value) { return "\"" + Convert.ToString(value, CultureInfo.InvariantCulture).Replace("\"", "\"\"") + "\""; }
        internal static string Full(string path) { return Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar); }
        internal static bool Within(string path, string directory)
        {
            string p = Full(path), d = Full(directory);
            return String.Equals(p, d, StringComparison.OrdinalIgnoreCase) || p.StartsWith(d + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
        }
        internal static Dictionary<string, object> Obj(params object[] values)
        {
            Dictionary<string, object> result = new Dictionary<string, object>();
            for (int i = 0; i < values.Length; i += 2) result[(string)values[i]] = values[i + 1]; return result;
        }
    }

    public sealed class EvidenceSession : IDisposable
    {
        private readonly object gate = new object();
        private readonly StreamWriter eventsWriter, csvWriter;
        private readonly string startedUtc;
        private readonly long startTicks;
        private readonly List<object> runs = new List<object>();
        private readonly List<object> errors = new List<object>();
        private readonly Dictionary<string, long> kindCounts = new Dictionary<string, long>(StringComparer.Ordinal);
        private readonly Dictionary<string, long> evidenceCounts = new Dictionary<string, long>(StringComparer.Ordinal) {
            { "inventory_snapshots", 0 }, { "operator_runs", 0 }, { "image_file_observations", 0 },
            { "appended_log_lines", 0 }, { "verified_sample_copies", 0 }, { "rdk_command_requests", 0 }
        };
        private long sequence;
        private bool closed;
        private string status = "recording";
        public string DirectoryPath { get; private set; }
        public string SessionId { get; private set; }
        public string RunId { get; private set; }
        public bool IsClosed { get { lock (gate) return closed; } }
        public EvidenceSession(string baseDirectory)
        {
            SessionId = DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ", CultureInfo.InvariantCulture) + "-" + Guid.NewGuid().ToString("N").Substring(0, 8);
            DirectoryPath = Path.Combine(Path.GetFullPath(baseDirectory), "dsx-session-" + SessionId);
            Directory.CreateDirectory(DirectoryPath);
            startedUtc = DateTime.UtcNow.ToString("o"); startTicks = Stopwatch.GetTimestamp();
            eventsWriter = new StreamWriter(new FileStream(Path.Combine(DirectoryPath, "events.jsonl"), FileMode.CreateNew, FileAccess.Write, FileShare.Read), new UTF8Encoding(false));
            try { csvWriter = new StreamWriter(new FileStream(Path.Combine(DirectoryPath, "timeline.csv"), FileMode.CreateNew, FileAccess.Write, FileShare.Read), new UTF8Encoding(true)); }
            catch { eventsWriter.Dispose(); throw; }
            eventsWriter.AutoFlush = true; csvWriter.AutoFlush = true;
            csvWriter.WriteLine("sequence,session_id,run_id,utc,monotonic_ticks,frequency,elapsed_ms,source,kind,request_id,capture_id,data_json");
            try
            {
                Add("collector", "session_started", Util.Obj("network_connections_opened_by_observer", 0, "clock", "Stopwatch + UTC", "schema_version", 1));
                SaveSummary("recording");
            }
            catch
            {
                try { eventsWriter.Dispose(); } finally { csvWriter.Dispose(); closed = true; }
                throw;
            }
        }
        public void NewRun(string name, IDictionary<string, object> parameters)
        {
            lock (gate)
            {
                RequireOpen(); RunId = Guid.NewGuid().ToString("N");
                Dictionary<string, object> run = Util.Obj("run_id", RunId, "name", name, "parameters", parameters, "source", "operator_entered", "started_utc", DateTime.UtcNow.ToString("o"));
                runs.Add(run); Add("operator", "run_started", run); SaveSummary(status);
            }
        }
        public void Add(string source, string kind, object data, string requestId = null, string captureId = null)
        {
            lock (gate)
            {
                RequireOpen(); long ticks = Stopwatch.GetTimestamp();
                Dictionary<string, object> item = Util.Obj("sequence", ++sequence, "session_id", SessionId, "run_id", RunId,
                    "utc", DateTime.UtcNow.ToString("o"), "monotonic_ticks", ticks, "frequency", Stopwatch.Frequency,
                    "elapsed_ms", (ticks - startTicks) * 1000.0 / Stopwatch.Frequency, "source", source, "kind", kind,
                    "request_id", requestId, "capture_id", captureId, "data", data);
                eventsWriter.WriteLine(Util.Json(item));
                object[] columns = { sequence, SessionId, RunId, item["utc"], ticks, Stopwatch.Frequency, item["elapsed_ms"], Util.Redact(source), Util.Redact(kind), Util.Redact(requestId), Util.Redact(captureId), Util.Json(data) };
                csvWriter.WriteLine(String.Join(",", columns.Select(v => Util.Csv(v)).ToArray()));
                // Count only after both event streams accepted the record. These are evidence
                // coverage counters, never a claim of motion, exposure or file completion.
                string key = kind ?? "(null)"; long count;
                kindCounts.TryGetValue(key, out count); kindCounts[key] = count + 1;
                if (source == "inventory" && kind == "snapshot") evidenceCounts["inventory_snapshots"]++;
                if (source == "operator" && kind == "run_started") evidenceCounts["operator_runs"]++;
                if (source == "log_tail" && kind == "log_line") evidenceCounts["appended_log_lines"]++;
                if (source == "sample_copy" && kind == "copy_verified") evidenceCounts["verified_sample_copies"]++;
                if (source == "rdk" && kind == "command_requested") evidenceCounts["rdk_command_requests"]++;
                if (source == "filesystem_poll" && (kind == "file_created_observed" || kind == "file_changed_observed"))
                {
                    IDictionary<string, object> fields = data as IDictionary<string, object>; object path;
                    if (fields != null && fields.TryGetValue("path", out path))
                    {
                        string extension = Path.GetExtension(Convert.ToString(path)).ToLowerInvariant();
                        if (new[] { ".tif", ".tiff", ".dsx", ".bmp", ".png", ".jpg", ".jpeg", ".jxr", ".wdp", ".hdp" }.Contains(extension)) evidenceCounts["image_file_observations"]++;
                    }
                }
            }
        }
        public void AddError(string area, Exception ex)
        {
            lock (gate)
            {
                Dictionary<string, object> error = Util.Obj("area", area, "type", ex.GetType().Name, "message", Util.Redact(ex.Message), "utc", DateTime.UtcNow.ToString("o"));
                errors.Add(error); Add("collector", "collection_error", error);
            }
        }
        public void SaveSummary(string newStatus)
        {
            lock (gate)
            {
                if (closed) return;
                status = newStatus;
                eventsWriter.Flush(); csvWriter.Flush();
                Dictionary<string, string> missingDescriptions = new Dictionary<string, string> {
                    { "inventory_snapshots", "未收集环境与文件盘点" }, { "operator_runs", "未填写并开始实验" },
                    { "image_file_observations", "未观察到新建或变化的图像文件；既有基线不计入" },
                    { "appended_log_lines", "未观察到追加的完整日志行" }, { "verified_sample_copies", "未取得复制校验通过的样本" },
                    { "rdk_command_requests", "未记录 RDK 命令请求；被动观察会话出现此项是正常情况，不表示失败" }
                };
                List<object> notCollected = new List<object>();
                foreach (KeyValuePair<string, long> count in evidenceCounts)
                    if (count.Value == 0) notCollected.Add(Util.Obj("category", count.Key, "reason", missingDescriptions[count.Key], "classification", "not_collected"));
                Util.WriteJson(Path.Combine(DirectoryPath, "summary.json"), Util.Obj("schema_version", 1, "session_id", SessionId,
                    "started_utc", startedUtc, "updated_utc", DateTime.UtcNow.ToString("o"), "status", status, "event_count", sequence,
                    "runs", runs, "errors", errors, "frequency", Stopwatch.Frequency,
                    "kind_counts", kindCounts, "evidence_counts", evidenceCounts, "not_collected", notCollected,
                    "coverage_note", "Counts describe successfully recorded evidence categories only. RDK requests are not proof of transmission/execution; image events and verified copies are not capture or position acceptance. Errors remain independent of missing categories.",
                    "interpretation", "Observed notifications, cached reported coordinates, file stability candidates and operator markers are not proof of motion or capture completion.",
                    "recovery", "Unterminated sessions retain readable JSONL. Preserve complete lines; a truncated last line is a gap, not an event."));
            }
        }
        private void RequireOpen() { if (closed) throw new InvalidOperationException("会话已封口，不能继续写入；请新建会话。"); }
        private void Seal()
        {
            if (closed) return;
            Exception failure = null;
            try
            {
                Add("collector", "recording_stopped", Util.Obj("meaning", "Evidence recording stopped; this is not a device stop."));
            SaveSummary(status == "recording" ? "recording_stopped" : status);
            }
            catch (Exception ex) { failure = ex; }
            finally
            {
                try { eventsWriter.Dispose(); } catch (Exception ex) { if (failure == null) failure = ex; }
                try { csvWriter.Dispose(); } catch (Exception ex) { if (failure == null) failure = ex; }
                closed = true;
            }
            if (failure != null) throw new IOException("会话封口写入失败，已有文件已保留，请检查磁盘并恢复导出。", failure);
        }
        public string ExportZip()
        {
            lock (gate)
            {
                Seal();
                List<object> files = new List<object>();
                foreach (string file in System.IO.Directory.GetFiles(DirectoryPath, "*", SearchOption.AllDirectories).OrderBy(p => p, StringComparer.OrdinalIgnoreCase))
                {
                    if (Path.GetFileName(file) == "manifest.json") continue;
                    files.Add(Util.Obj("path", file.Substring(DirectoryPath.Length + 1).Replace('\\', '/'), "length", new FileInfo(file).Length, "sha256", Util.Sha256(file)));
                }
                Util.WriteJson(Path.Combine(DirectoryPath, "manifest.json"), Util.Obj("schema_version", 1, "session_id", SessionId, "files", files, "self_hash", "Manifest is excluded from itself; ZIP has an external SHA256 sidecar."));
                string zip = DirectoryPath + "-" + DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ") + "-" + Guid.NewGuid().ToString("N").Substring(0, 6) + ".zip";
                try
                {
                    ZipFile.CreateFromDirectory(DirectoryPath, zip, CompressionLevel.Optimal, false);
                    File.WriteAllText(zip + ".sha256", Util.Sha256(zip) + "  " + Path.GetFileName(zip) + Environment.NewLine, new UTF8Encoding(false));
                    return zip;
                }
                catch (Exception ex)
                {
                    Util.WriteJson(Path.Combine(DirectoryPath, "export-error-" + Guid.NewGuid().ToString("N") + ".json"), Util.Obj("utc", DateTime.UtcNow.ToString("o"), "message", ex.Message, "partial_zip", zip));
                    throw;
                }
            }
        }
        public static Dictionary<string, object> InspectRecovery(string directory)
        {
            // Read-only inspection: never repairs or overwrites the original partial session.
            string full = Path.GetFullPath(directory); long valid = 0; List<object> gaps = new List<object>();
            try
            {
                using (StreamReader reader = new StreamReader(Path.Combine(full, "events.jsonl"), Encoding.UTF8))
                {
                    string line; long number = 0;
                    while ((line = reader.ReadLine()) != null)
                    {
                        number++;
                        try
                        {
                            Dictionary<string, object> item = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(line);
                            if (item == null || !item.ContainsKey("sequence") || !item.ContainsKey("session_id")) throw new InvalidDataException("事件字段缺失。");
                            valid++;
                        }
                        catch (Exception ex) { gaps.Add(Util.Obj("line", number, "error", ex.GetType().Name, "action", "Preserve original; do not count this line as an event.")); }
                    }
                }
            }
            catch (Exception ex) { gaps.Add(Util.Obj("area", "events", "error", ex.Message)); }
            try { new JavaScriptSerializer().DeserializeObject(File.ReadAllText(Path.Combine(full, "summary.json"))); }
            catch (Exception ex) { gaps.Add(Util.Obj("area", "summary", "error", ex.GetType().Name)); }
            return Util.Obj("directory", full, "valid_event_lines", valid, "gaps", gaps, "read_only", true, "complete", gaps.Count == 0,
                "meaning", "Readable evidence only; this does not infer a previous device operation outcome.");
        }
        public void Dispose() { lock (gate) Seal(); }
    }

    public sealed class InventoryResult
    {
        public string Root;
        public bool Matched;
        public Dictionary<string, object> Data;
    }
    public static class Inventory
    {
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool IsWow64Process(IntPtr process, out bool wow64);
        public static string DiscoverRoot()
        {
            string[] roots = { Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), @"C:\OLYMPUS", @"C:\Program Files\OLYMPUS", @"C:\Program Files (x86)\OLYMPUS" };
            foreach (string root in roots.Where(p => !String.IsNullOrWhiteSpace(p)))
                foreach (string candidate in new[] { Path.Combine(root, "OLYMPUS", "DSX-BSW"), Path.Combine(root, "DSX-BSW"), root })
                    if (Directory.Exists(Path.Combine(candidate, "DSX-BSW-CTRL"))) return candidate;
            return "";
        }
        public static InventoryResult Capture(string root, string baselinePath)
        {
            List<object> gaps = new List<object>(), identities = new List<object>(), processes = new List<object>(), listeners = new List<object>();
            HashSet<int> dsxPids = new HashSet<int>();
            Dictionary<string, object> data = Util.Obj("captured_utc", DateTime.UtcNow.ToString("o"), "os_reported", Environment.OSVersion.VersionString,
                "os_64bit", Environment.Is64BitOperatingSystem, "collector_64bit", Environment.Is64BitProcess, "clr", Environment.Version.ToString(),
                "culture", CultureInfo.CurrentCulture.Name, "ui_culture", CultureInfo.CurrentUICulture.Name, "timezone", TimeZoneInfo.Local.Id,
                "identities", identities, "processes", processes, "listeners", listeners, "gaps", gaps, "root", root,
                "privacy", "Process command lines are never queried. Only DSX process metadata is collected.", "network_mode", "passive local TCP listener table; no connect or port probe");
            try
            {
                using (RegistryKey os = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Windows NT\CurrentVersion"))
                    data["windows_registry"] = Util.Obj("product_name", os == null ? null : os.GetValue("ProductName"), "release_id", os == null ? null : os.GetValue("ReleaseId"), "display_version", os == null ? null : os.GetValue("DisplayVersion"), "build", os == null ? null : os.GetValue("CurrentBuildNumber"), "ubr", os == null ? null : os.GetValue("UBR"));
                using (RegistryKey dotnet = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full")) data["dotnet_release"] = dotnet == null ? null : dotnet.GetValue("Release");
            }
            catch (Exception ex) { gaps.Add(Util.Obj("area", "registry", "error", ex.Message)); }
            bool matched = true; int compared = 0;
            try
            {
                if (String.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) throw new DirectoryNotFoundException("未找到 DSX-BSW 根目录。");
                Dictionary<string, object> baseline = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(File.ReadAllText(baselinePath));
                IEnumerable rows = baseline["files"] as IEnumerable;
                if (rows == null) throw new InvalidDataException("基线 files 缺失。");
                foreach (object row in rows)
                {
                    IDictionary<string, object> item = row as IDictionary<string, object>;
                    if (item == null) throw new InvalidDataException("基线文件项无效。");
                    string relative = Convert.ToString(item["path"]), expected = Convert.ToString(item["sha256"]);
                    string full = Path.GetFullPath(Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar)));
                    if (!Util.Within(full, root) || Path.IsPathRooted(relative)) throw new InvalidDataException("基线路径越界。");
                    Dictionary<string, object> identity = Util.Obj("path", relative, "expected_sha256", expected, "matched", false);
                    try
                    {
                        FileInfo before = new FileInfo(full); long beforeLength = before.Length; DateTime beforeWrite = before.LastWriteTimeUtc;
                        string hash = Util.Sha256(full); before.Refresh();
                        bool stable = beforeLength == before.Length && beforeWrite == before.LastWriteTimeUtc;
                        FileVersionInfo version = FileVersionInfo.GetVersionInfo(full);
                        bool equal = stable && String.Equals(hash, expected, StringComparison.OrdinalIgnoreCase);
                        identity["sha256"] = hash; identity["length"] = before.Length; identity["file_version"] = version.FileVersion; identity["product_version"] = version.ProductVersion;
                        identity["stable_during_hash"] = stable; identity["matched"] = equal; if (!equal) matched = false;
                    }
                    catch (Exception ex) { identity["error"] = ex.Message; matched = false; gaps.Add(Util.Obj("area", "identity", "path", relative, "error", ex.Message)); }
                    identities.Add(identity); compared++;
                }
            }
            catch (Exception ex) { matched = false; gaps.Add(Util.Obj("area", "baseline", "error", ex.Message)); }
            foreach (Process process in Process.GetProcesses())
            {
                using (process)
                {
                    try
                    {
                        if (process.ProcessName.IndexOf("DSX", StringComparison.OrdinalIgnoreCase) < 0) continue;
                        dsxPids.Add(process.Id);
                        Dictionary<string, object> info = Util.Obj("pid", process.Id, "name", process.ProcessName);
                        try { info["path"] = process.MainModule.FileName; } catch (Exception ex) { info["path_gap"] = ex.Message; }
                        try { info["started_utc"] = process.StartTime.ToUniversalTime().ToString("o"); } catch (Exception ex) { info["start_gap"] = ex.Message; }
                        try { bool wow64; if (!IsWow64Process(process.Handle, out wow64)) throw new System.ComponentModel.Win32Exception(); info["bitness"] = Environment.Is64BitOperatingSystem && !wow64 ? 64 : 32; } catch (Exception ex) { info["bitness_gap"] = ex.Message; }
                        try
                        {
                            List<object> modules = new List<object>();
                            foreach (ProcessModule module in process.Modules) modules.Add(Util.Obj("name", module.ModuleName, "path", module.FileName));
                            info["modules"] = modules;
                        }
                        catch (Exception ex) { info["module_gap"] = ex.Message; }
                        processes.Add(info);
                    }
                    catch (Exception ex) { gaps.Add(Util.Obj("area", "process_snapshot", "error", ex.Message)); }
                }
            }
            try { foreach (System.Net.IPEndPoint endpoint in IPGlobalProperties.GetIPGlobalProperties().GetActiveTcpListeners().Where(e => e.Port == 35001)) listeners.Add(Util.Obj("address", endpoint.Address.ToString(), "port", endpoint.Port, "owner_pid", null)); }
            catch (Exception ex) { gaps.Add(Util.Obj("area", "tcp_listener_table", "error", ex.Message)); }
            data["tcp_ownership_and_connections"] = CaptureTcpOwnership(dsxPids, gaps);
            if (compared == 0) matched = false;
            data["identity_matched"] = matched; data["compared_file_count"] = compared;
            data["listener_owner_note"] = "Port 35001 listeners from IPGlobalProperties plus bounded system netstat TCP/PID snapshot filtered to port 35001 or observed DSX PIDs. Established rows may identify another client; race between snapshots remains possible. No remote host is contacted.";
            return new InventoryResult { Root = root, Matched = matched, Data = data };
        }
        private static List<object> CaptureTcpOwnership(HashSet<int> dsxPids, List<object> gaps)
        {
            List<object> selected = new List<object>(); List<string> lines = new List<string>(); StringBuilder stderr = new StringBuilder();
            try
            {
                string executable = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "netstat.exe");
                if (!File.Exists(executable)) throw new FileNotFoundException("系统 netstat.exe 不可用。", executable);
                using (Process process = new Process())
                {
                    int outputDone = 0, errorDone = 0;
                    process.StartInfo = new ProcessStartInfo(executable, "-ano") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
                    process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data == null) Interlocked.Exchange(ref outputDone, 1); else lock (lines) { if (lines.Count < 20000) lines.Add(e.Data); } };
                    process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data == null) Interlocked.Exchange(ref errorDone, 1); else lock (stderr) { if (stderr.Length < 4096) stderr.AppendLine(e.Data); } };
                    process.Start(); process.BeginOutputReadLine(); process.BeginErrorReadLine();
                    if (!process.WaitForExit(5000))
                    {
                        try { process.Kill(); process.WaitForExit(1000); } catch (Exception stopError) { gaps.Add(Util.Obj("area", "netstat_timeout_cleanup", "error", stopError.Message)); }
                        throw new TimeoutException("被动 netstat 表采集超过 5 秒；结果不完整。");
                    }
                    // Bound the callback drain too; do not use the unbounded WaitForExit overload.
                    Stopwatch drain = Stopwatch.StartNew();
                    while ((Interlocked.CompareExchange(ref outputDone, 0, 0) == 0 || Interlocked.CompareExchange(ref errorDone, 0, 0) == 0) && drain.ElapsedMilliseconds < 1000) Thread.Sleep(10);
                    if (outputDone == 0 || errorDone == 0) gaps.Add(Util.Obj("area", "netstat_output_drain", "error", "Process exited but output callbacks did not finish within 1 second; snapshot may be incomplete."));
                    string errorText; lock (stderr) errorText = stderr.ToString();
                    if (process.ExitCode != 0) throw new IOException("netstat 返回 " + process.ExitCode + ": " + Util.Redact(errorText));
                }
                List<string> snapshot; lock (lines) snapshot = new List<string>(lines);
                if (snapshot.Count >= 20000) gaps.Add(Util.Obj("area", "netstat_output_limit", "error", "Output limited to 20000 lines; TCP snapshot may be incomplete."));
                foreach (string line in snapshot)
                {
                    string[] parts = Regex.Split(line.Trim(), @"\s+");
                    if (parts.Length < 5 || !String.Equals(parts[0], "TCP", StringComparison.OrdinalIgnoreCase)) continue;
                    int pid; if (!Int32.TryParse(parts[parts.Length - 1], out pid)) continue;
                    int localPort = EndpointPort(parts[1]), remotePort = EndpointPort(parts[2]);
                    if (localPort != 35001 && remotePort != 35001 && !dsxPids.Contains(pid)) continue;
                    selected.Add(Util.Obj("source", "system_netstat_passive", "local_endpoint", parts[1], "remote_endpoint", parts[2],
                        "state", parts[3], "owner_pid", pid, "owner_is_observed_dsx", dsxPids.Contains(pid), "local_port", localPort, "remote_port", remotePort,
                        "observed_utc", DateTime.UtcNow.ToString("o")));
                }
            }
            catch (Exception ex) { gaps.Add(Util.Obj("area", "tcp_ownership_and_connections", "error", ex.Message)); }
            return selected;
        }
        private static int EndpointPort(string endpoint)
        {
            int colon = endpoint.LastIndexOf(':'); int value;
            return colon >= 0 && Int32.TryParse(endpoint.Substring(colon + 1), NumberStyles.None, CultureInfo.InvariantCulture, out value) ? value : -1;
        }
    }

    public sealed class ObservationRecorder : IDisposable
    {
        private sealed class Fingerprint
        {
            public long Length; public long WriteTicks; public long CreationTicks;
            public static Fingerprint Read(string path) { FileInfo i = new FileInfo(path); return new Fingerprint { Length = i.Length, WriteTicks = i.LastWriteTimeUtc.Ticks, CreationTicks = i.CreationTimeUtc.Ticks }; }
            public bool Same(Fingerprint other) { return other != null && Length == other.Length && WriteTicks == other.WriteTicks && CreationTicks == other.CreationTicks; }
            public object Data() { return Util.Obj("length", Length, "mtime_utc_ticks", WriteTicks, "creation_utc_ticks", CreationTicks); }
        }
        private sealed class FileState { public Fingerprint Fingerprint; public long LastChange; public bool Stable; public bool Log; }
        private sealed class TailState
        {
            public long Offset; public long Creation; public string Boundary; public Encoding Encoding; public Decoder Decoder;
            public StringBuilder Pending = new StringBuilder(); public bool DroppingLongLine; public bool Baseline; public bool SensitiveXml;
        }
        private readonly EvidenceSession session;
        private readonly string imageDirectory, logDirectory;
        private readonly object gate = new object();
        private readonly Dictionary<string, FileState> states = new Dictionary<string, FileState>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, TailState> tails = new Dictionary<string, TailState>(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> observed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> reportedErrors = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private readonly List<FileSystemWatcher> watchers = new List<FileSystemWatcher>();
        private Timer timer; private bool running; private long lastPoll;
        public string LastError { get; private set; }
        public ObservationRecorder(EvidenceSession session, string imageDirectory, string logDirectory)
        {
            this.session = session; this.imageDirectory = Normalize(imageDirectory); this.logDirectory = Normalize(logDirectory);
            foreach (string root in new[] { this.imageDirectory, this.logDirectory }.Where(p => p != null))
                if (Util.Within(root, session.DirectoryPath) || Util.Within(session.DirectoryPath, root)) throw new ArgumentException("观察目录不能包含或位于本次证据目录中。");
        }
        private static string Normalize(string path) { return String.IsNullOrWhiteSpace(path) ? null : Util.Full(path); }
        public List<string> ObservedFiles { get { lock (gate) return observed.OrderBy(p => p, StringComparer.OrdinalIgnoreCase).ToList(); } }
        public void Start()
        {
            lock (gate)
            {
                if (running) return;
                if (imageDirectory == null && logDirectory == null) throw new ArgumentException("至少选择一个图像或日志观察目录。");
                running = true; lastPoll = Stopwatch.GetTimestamp();
                session.Add("observer", "observation_started", Util.Obj("image_directory", imageDirectory, "log_directory", logDirectory, "poll_interval_ms", 500, "stable_threshold_ms", 2000,
                    "limits", "Polling timestamps are observation times, not vendor event times. Same-size rewrites between polls may be missed. Existing logs start at EOF. FSW rename notifications supplement polling."));
                Poll(true);
                foreach (string root in new[] { imageDirectory, logDirectory }.Where(p => p != null).Distinct(StringComparer.OrdinalIgnoreCase))
                {
                    try
                    {
                        FileSystemWatcher watcher = new FileSystemWatcher(root) { IncludeSubdirectories = true, NotifyFilter = NotifyFilters.FileName | NotifyFilters.DirectoryName };
                        watcher.Renamed += delegate(object sender, RenamedEventArgs e) { RecordRename(e); };
                        watcher.Error += delegate(object sender, ErrorEventArgs e) { RecordWatcherError(e); };
                        watchers.Add(watcher); watcher.EnableRaisingEvents = true;
                    }
                    catch (Exception ex) { ReportOnce("watcher:" + root, ex); }
                }
                timer = new Timer(delegate { Tick(); }, null, 500, 500);
            }
        }
        private void RecordRename(RenamedEventArgs e)
        {
            lock (gate)
            {
                if (!running) return;
                try { session.Add("filesystem_watcher", "rename_observed", Util.Obj("old_path", e.OldFullPath, "new_path", e.FullPath, "meaning", "OS notification only; file identity and completeness are not established.")); }
                catch (Exception ex) { BackgroundFailure(ex); }
            }
        }
        private void RecordWatcherError(ErrorEventArgs e)
        {
            lock (gate)
            {
                if (!running) return;
                try { ReportOnce("filesystem_watcher", e.GetException()); } catch (Exception ex) { BackgroundFailure(ex); }
            }
        }
        private void BackgroundFailure(Exception ex)
        {
            LastError = Util.Redact(ex.ToString());
            try { ReportOnce("observation_background", ex); }
            catch (Exception writeError)
            {
                // A full/disconnected evidence disk must not terminate the GUI process from
                // a ThreadPool callback. The persistent LastError is displayed by the GUI.
                LastError += "\n证据写入失败：" + Util.Redact(writeError.Message);
                running = false; if (timer != null) { timer.Dispose(); timer = null; }
            }
        }
        private void Tick()
        {
            lock (gate)
            {
                if (!running) return;
                try { Poll(false); }
                catch (Exception ex) { BackgroundFailure(ex); }
            }
        }
        private void ReportOnce(string area, Exception ex) { if (reportedErrors.Add(area + ":" + ex.GetType().Name + ":" + ex.Message)) session.AddError(area, ex); }
        private IEnumerable<string> Enumerate(string root)
        {
            if (root == null) yield break;
            Stack<string> directories = new Stack<string>(); directories.Push(root); int count = 0;
            while (directories.Count > 0)
            {
                string current = directories.Pop(); string[] files = new string[0], subdirs = new string[0];
                try
                {
                    if (Util.Within(current, session.DirectoryPath) || (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0) continue;
                    files = Directory.GetFiles(current); subdirs = Directory.GetDirectories(current);
                }
                catch (Exception ex) { ReportOnce("enumerate:" + current, ex); }
                foreach (string subdir in subdirs) directories.Push(subdir);
                foreach (string file in files)
                {
                    if (++count > 100000) { ReportOnce("file_limit:" + root, new IOException("单目录观察超过 100000 文件，后续文件未枚举。")); yield break; }
                    if ((File.GetAttributes(file) & FileAttributes.ReparsePoint) == 0) yield return file;
                }
            }
        }
        private void Poll(bool baseline)
        {
            long now = Stopwatch.GetTimestamp(); double elapsed = (now - lastPoll) * 1000.0 / Stopwatch.Frequency; lastPoll = now;
            if (!baseline && elapsed > 1000) session.Add("observer", "poll_gap", Util.Obj("actual_interval_ms", elapsed, "nominal_interval_ms", 500));
            HashSet<string> found = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (string path in Enumerate(imageDirectory).Concat(Enumerate(logDirectory)))
            {
                if (!found.Add(path)) continue;
                try
                {
                    Fingerprint fp = Fingerprint.Read(path); bool isLog = logDirectory != null && Util.Within(path, logDirectory);
                    FileState state;
                    if (!states.TryGetValue(path, out state))
                    {
                        state = new FileState { Fingerprint = fp, LastChange = now, Log = isLog }; states[path] = state;
                        session.Add("filesystem_poll", baseline ? "baseline_file" : "file_created_observed", Util.Obj("path", path, "fingerprint", fp.Data(), "is_log", isLog));
                        // Baseline files remain selectable but are explicitly distinguished from newly created images.
                        observed.Add(path);
                    }
                    else if (!state.Fingerprint.Same(fp))
                    {
                        session.Add("filesystem_poll", "file_changed_observed", Util.Obj("path", path, "before", state.Fingerprint.Data(), "after", fp.Data()));
                        state.Fingerprint = fp; state.LastChange = now; state.Stable = false;
                    }
                    if (!baseline && !state.Stable && (now - state.LastChange) * 1000.0 / Stopwatch.Frequency >= 2000)
                    {
                        state.Stable = true; session.Add("filesystem_poll", "stable_candidate", Util.Obj("path", path, "fingerprint", fp.Data(), "meaning", "Unchanged metadata over polling window; not capture, write-close or image integrity proof."));
                    }
                    if (isLog && IsTextLog(path)) Tail(path, fp, baseline);
                }
                catch (Exception ex) { ReportOnce("observe:" + path, ex); }
            }
            foreach (string deleted in states.Keys.Where(p => !found.Contains(p)).ToList())
            {
                session.Add("filesystem_poll", "file_missing_observed", Util.Obj("path", deleted, "meaning", "May be deleted, moved, inaccessible, or omitted after an enumeration failure."));
                FlushTail(deleted, "log_removed_partial"); states.Remove(deleted); tails.Remove(deleted);
            }
        }
        private static bool IsTextLog(string path) { string ext = Path.GetExtension(path).ToLowerInvariant(); return ext == ".log" || ext == ".txt" || ext == ".csv" || ext == ".xml" || ext == ".json" || ext == ".ini"; }
        private static bool IsImageSample(string path)
        {
            string extension = Path.GetExtension(path).ToLowerInvariant();
            return new[] { ".tif", ".tiff", ".dsx", ".bmp", ".png", ".jpg", ".jpeg", ".jxr", ".wdp", ".hdp" }.Contains(extension);
        }
        private static FileStream OpenShared(string path) { return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete); }
        private static Encoding DetectEncoding(FileStream stream, out int preamble)
        {
            byte[] bytes = new byte[4096]; int n = stream.Read(bytes, 0, bytes.Length); stream.Position = 0; preamble = 0;
            if (n >= 3 && bytes[0] == 239 && bytes[1] == 187 && bytes[2] == 191) { preamble = 3; return new UTF8Encoding(false, true); }
            if (n >= 2 && bytes[0] == 255 && bytes[1] == 254) { preamble = 2; return new UnicodeEncoding(false, false, true); }
            if (n >= 2 && bytes[0] == 254 && bytes[1] == 255) { preamble = 2; return new UnicodeEncoding(true, false, true); }
            try { new UTF8Encoding(false, true).GetString(bytes, 0, n); return new UTF8Encoding(false, true); }
            catch (DecoderFallbackException) { return Encoding.GetEncoding(CultureInfo.CurrentCulture.TextInfo.ANSICodePage); }
        }
        private static string Boundary(FileStream stream, long offset)
        {
            long position = stream.Position; long start = Math.Max(0, offset - 128); stream.Position = start;
            byte[] bytes = new byte[(int)Math.Max(0, Math.Min(128, offset - start))]; int n = stream.Read(bytes, 0, bytes.Length); stream.Position = position;
            return Convert.ToBase64String(bytes, 0, n);
        }
        private void Tail(string path, Fingerprint fp, bool baseline)
        {
            using (FileStream stream = OpenShared(path))
            {
                TailState tail; bool reset = false;
                if (tails.TryGetValue(path, out tail)) reset = fp.Length < tail.Offset || fp.CreationTicks != tail.Creation || Boundary(stream, Math.Min(tail.Offset, stream.Length)) != tail.Boundary;
                if (tail == null || reset)
                {
                    if (reset) { FlushTail(path, "log_rotation_partial"); session.Add("log_tail", "log_reset_observed", Util.Obj("path", path, "reason", "Length, creation time or prior read-boundary changed.")); }
                    int preamble; Encoding encoding = DetectEncoding(stream, out preamble);
                    tail = new TailState { Offset = baseline && !reset ? stream.Length : preamble, Creation = fp.CreationTicks, Encoding = encoding, Decoder = encoding.GetDecoder(), Baseline = baseline };
                    // A log may already end halfway through /PASS:secret. Never emit the
                    // remainder of an existing partial line without its redaction prefix.
                    if (baseline && !reset && stream.Length > preamble)
                    {
                        long offset = Math.Max(preamble, stream.Length - 4); stream.Position = offset;
                        byte[] last = new byte[(int)(stream.Length - offset)]; stream.Read(last, 0, last.Length);
                        string end;
                        try { end = encoding.GetString(last); } catch (DecoderFallbackException) { end = "?"; }
                        tail.DroppingLongLine = !end.EndsWith("\n", StringComparison.Ordinal);
                        if (tail.DroppingLongLine) session.Add("log_tail", "baseline_partial_line_skipped", Util.Obj("path", path));
                    }
                    tails[path] = tail;
                    session.Add("log_tail", "log_opened", Util.Obj("path", path, "encoding", encoding.WebName, "start_offset", tail.Offset, "baseline_skipped", baseline && !reset, "encoding_detection", "BOM, otherwise UTF-8 probe, otherwise current ANSI code page; verify on site."));
                }
                stream.Position = Math.Min(tail.Offset, stream.Length);
                byte[] bytes = new byte[65536]; int count = stream.Read(bytes, 0, bytes.Length);
                if (count > 0)
                {
                    char[] chars = new char[tail.Encoding.GetMaxCharCount(count)]; int length;
                    try { length = tail.Decoder.GetChars(bytes, 0, count, chars, 0, false); }
                    catch (DecoderFallbackException)
                    {
                        session.Add("log_tail", "log_encoding_gap", Util.Obj("path", path, "offset", tail.Offset, "byte_count_skipped", count));
                        tail.Decoder = tail.Encoding.GetDecoder(); tail.Pending.Clear(); tail.DroppingLongLine = true; length = 0;
                    }
                    for (int i = 0; i < length; i++)
                    {
                        char ch = chars[i];
                        if (ch == '\n')
                        {
                            if (!tail.DroppingLongLine) session.Add("log_tail", "log_line", Util.Obj("path", path, "text", Util.RedactLogLine(tail.Pending.ToString().TrimEnd('\r'), ref tail.SensitiveXml), "timestamp_kind", "collector_observed"));
                            tail.Pending.Clear(); tail.DroppingLongLine = false;
                        }
                        else if (!tail.DroppingLongLine)
                        {
                            tail.Pending.Append(ch);
                            if (tail.Pending.Length > 32768) { tail.Pending.Clear(); tail.DroppingLongLine = true; session.Add("log_tail", "oversized_log_line_skipped", Util.Obj("path", path, "limit_characters", 32768)); }
                        }
                    }
                    tail.Offset += count;
                }
                tail.Boundary = Boundary(stream, tail.Offset);
            }
        }
        private void FlushTail(string path, string kind)
        {
            TailState tail;
            if (tails.TryGetValue(path, out tail) && tail.Pending.Length > 0)
            { session.Add("log_tail", kind, Util.Obj("path", path, "characters_skipped", tail.Pending.Length, "complete_line", false, "reason", "Incomplete lines are omitted to avoid leaking a split credential.")); tail.Pending.Clear(); }
        }
        public void Stop()
        {
            lock (gate)
            {
                if (!running && timer == null && watchers.Count == 0) return;
                if (timer != null) { timer.Dispose(); timer = null; }
                try { Poll(false); } catch (Exception ex) { BackgroundFailure(ex); }
                running = false;
                foreach (FileSystemWatcher watcher in watchers) watcher.Dispose(); watchers.Clear();
                try
                {
                    foreach (string path in tails.Keys.ToList()) FlushTail(path, "log_final_partial");
                    session.Add("observer", "observation_stopped", Util.Obj("meaning", "Stops evidence observation only; no device command is sent."));
                }
                catch (Exception ex) { BackgroundFailure(ex); }
            }
        }
        public void CopySamples(IEnumerable<string> paths)
        {
            lock (gate)
            {
                if (running) throw new InvalidOperationException("先停止观察，再复制并校验样本，避免影响测量。");
                if (session.IsClosed) throw new InvalidOperationException("会话已封口，不能再加入样本。");
                string destination = Path.Combine(session.DirectoryPath, "samples"); Directory.CreateDirectory(destination);
                foreach (string selected in paths.Distinct(StringComparer.OrdinalIgnoreCase))
                {
                    string path = Path.GetFullPath(selected);
                    if (!observed.Contains(path)) { session.AddError("sample_selection", new InvalidOperationException("样本未在本次观察清单中：" + path)); continue; }
                    string target = Path.Combine(destination, Guid.NewGuid().ToString("N").Substring(0, 10) + "-" + Path.GetFileName(path));
                    try
                    {
                        if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new IOException("拒绝复制符号链接/重解析点样本。");
                        bool log = logDirectory != null && Util.Within(path, logDirectory);
                        if (log && !IsTextLog(path)) throw new IOException("日志目录仅允许脱敏文本日志副本（log/txt/csv/xml/json/ini）。");
                        if (!log && !IsImageSample(path)) throw new IOException("图像目录仅允许图像样本（tif/tiff/dsx/bmp/png/jpg/jpeg/jxr/wdp/hdp），其它文件不能原样复制。");
                        Fingerprint before = Fingerprint.Read(path); string hashBefore = Util.Sha256(path);
                        FileState lastObserved; bool matchesObserved = states.TryGetValue(path, out lastObserved) && lastObserved.Fingerprint.Same(before);
                        if (log)
                        {
                            using (FileStream input = OpenShared(path))
                            {
                                int preamble; Encoding encoding = DetectEncoding(input, out preamble); input.Position = preamble;
                                using (StreamReader reader = new StreamReader(input, encoding, true))
                                using (StreamWriter output = new StreamWriter(new FileStream(target, FileMode.CreateNew, FileAccess.Write), new UTF8Encoding(false)))
                                { string line; bool sensitiveXml = false; while ((line = reader.ReadLine()) != null) output.WriteLine(Util.RedactLogLine(line, ref sensitiveXml)); }
                            }
                        }
                        else { using (FileStream input = OpenShared(path)) using (FileStream output = new FileStream(target, FileMode.CreateNew, FileAccess.Write)) input.CopyTo(output); }
                        string copiedHash = Util.Sha256(target), hashAfter = Util.Sha256(path); Fingerprint after = Fingerprint.Read(path);
                        bool verified = matchesObserved && before.Same(after) && hashBefore == hashAfter && (log || copiedHash == hashBefore);
                        session.Add("sample_copy", verified ? "copy_verified" : "copy_unverified_source_changed", Util.Obj("source_path", path,
                            "destination", target.Substring(session.DirectoryPath.Length + 1), "source_before", before.Data(), "source_after", after.Data(),
                            "source_sha256_before", hashBefore, "source_sha256_after", hashAfter, "copied_sha256", copiedHash,
                            "sanitized_text_copy", log, "matches_last_observed_metadata", matchesObserved, "meaning", log ? "Stable source was decoded and redacted; copied hash identifies sanitized UTF-8 text, not the source bytes." : "Byte-copy verification does not establish image/position correspondence."));
                    }
                    catch (Exception ex) { session.AddError("copy:" + path, ex); }
                }
            }
        }
        public void Dispose() { Stop(); }
    }
}
