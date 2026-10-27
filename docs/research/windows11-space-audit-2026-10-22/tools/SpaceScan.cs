// C# 5 / .NET Framework. Metadata-only Windows C: audit. No deletion or file-content reads.
// PowerShell: Add-Type -Path .\SpaceScan.cs; [SpaceScan]::Run('C:\', 'C:\tmp\space-audit-unique')
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

public static class SpaceScan
{
    private const long LargeThreshold = 16L * 1024 * 1024;
    private const uint OpenExisting = 3, ShareAll = 7, OpenReparsePoint = 0x00200000, BackupSemantics = 0x02000000;
    [StructLayout(LayoutKind.Sequential)] private struct NativeFileTime { public uint Low, High; }
    [StructLayout(LayoutKind.Sequential)] private struct ByHandleInfo
    {
        public uint Attributes; public NativeFileTime Created, Accessed, Written;
        public uint VolumeSerial, SizeHigh, SizeLow, Links, IndexHigh, IndexLow;
    }
    [StructLayout(LayoutKind.Sequential)] private struct StandardInfo
    {
        public long AllocationSize, EndOfFile; public uint NumberOfLinks;
        [MarshalAs(UnmanagedType.U1)] public bool DeletePending;
        [MarshalAs(UnmanagedType.U1)] public bool Directory;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern SafeFileHandle CreateFile(string name, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetFileInformationByHandle(SafeFileHandle handle, out ByHandleInfo info);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetFileInformationByHandleEx(SafeFileHandle handle, int infoClass, out StandardInfo info, uint length);

    private struct FileKey : IEquatable<FileKey>
    {
        public uint Volume; public ulong Id;
        public bool Equals(FileKey other) { return Volume == other.Volume && Id == other.Id; }
        public override bool Equals(object other) { return other is FileKey && Equals((FileKey)other); }
        public override int GetHashCode() { return unchecked((int)Volume ^ (int)Id ^ (int)(Id >> 32)); }
        public override string ToString() { return Volume.ToString("x8") + ":" + Id.ToString("x16"); }
    }
    private sealed class Totals
    {
        public long Files, Directories, LogicalKnownBytes, AllocationPathKnownBytes, UniqueLogicalAssignedBytes, UniqueAllocationAssignedBytes;
        public long UnknownLogicalFiles, UnknownAllocationFiles, UnknownIdentityFiles, DuplicateIdentityPaths, ReparseSkipped, Errors, OutputExcluded;
        public void Add(Totals o)
        {
            Files += o.Files; Directories += o.Directories; LogicalKnownBytes += o.LogicalKnownBytes; AllocationPathKnownBytes += o.AllocationPathKnownBytes;
            UniqueLogicalAssignedBytes += o.UniqueLogicalAssignedBytes; UniqueAllocationAssignedBytes += o.UniqueAllocationAssignedBytes;
            UnknownLogicalFiles += o.UnknownLogicalFiles; UnknownAllocationFiles += o.UnknownAllocationFiles; UnknownIdentityFiles += o.UnknownIdentityFiles;
            DuplicateIdentityPaths += o.DuplicateIdentityPaths; ReparseSkipped += o.ReparseSkipped; Errors += o.Errors; OutputExcluded += o.OutputExcluded;
        }
        public Dictionary<string, object> Data()
        {
            return Obj("files", Files, "directories_including_self", Directories, "logical_known_bytes", LogicalKnownBytes,
                "allocation_path_known_bytes", AllocationPathKnownBytes, "unique_logical_assigned_bytes", UniqueLogicalAssignedBytes,
                "unique_allocation_assigned_bytes", UniqueAllocationAssignedBytes, "unknown_logical_files", UnknownLogicalFiles,
                "unknown_allocation_files", UnknownAllocationFiles, "unknown_identity_files", UnknownIdentityFiles,
                "duplicate_identity_paths", DuplicateIdentityPaths, "reparse_skipped", ReparseSkipped, "errors", Errors, "output_excluded", OutputExcluded);
        }
    }
    private sealed class Frame
    {
        public string Path; public int Depth; public IEnumerator<string> Entries; public bool Started;
        public Totals Direct = new Totals(), Recursive = new Totals { Directories = 1 };
    }
    private sealed class FileRow
    {
        public string Path, Id, CreatedUtc, ModifiedUtc; public uint Attributes; public long? Logical, Allocation; public uint? Links;
        public bool? Duplicate; public bool FirstLogical, FirstAllocation; public string Error;
        public object Data() { return Obj("path", Path, "logical_bytes", Logical, "allocation_bytes", Allocation, "links", Links, "file_id", Id,
            "duplicate_identity", Duplicate, "first_logical_assignment", FirstLogical, "first_allocation_assignment", FirstAllocation,
            "created_utc", CreatedUtc, "modified_utc", ModifiedUtc, "attributes", Attributes, "error", Error); }
        public object[] Columns() { return new object[] { Path, Logical, Allocation, Links, Id, Duplicate, FirstLogical, FirstAllocation, CreatedUtc, ModifiedUtc, Attributes, Error }; }
    }
    private sealed class Scan
    {
        public string Root, Output, StartedUtc;
        public Stopwatch Watch = Stopwatch.StartNew(); public long LastProgressMs;
        public StreamWriter Index, DirectoriesCsv, DirectoriesJson, Large, Errors, Reparse;
        public Dictionary<FileKey, byte> Seen = new Dictionary<FileKey, byte>();
        public List<FileRow> Top = new List<FileRow>();
        public Totals Progress = new Totals(); public long DirectoryRows, LargeRows; public bool FirstDirectory = true;
        public void Error(Frame frame, string path, string operation, string message, int? code)
        {
            WriteCsv(Errors, new object[] { DateTime.UtcNow.ToString("o"), path, operation, code, message });
            frame.Direct.Errors++; frame.Recursive.Errors++; Progress.Errors++;
        }
        public void ProgressFile(string current, bool force)
        {
            if (!force && Progress.Files % 5000 != 0 && Watch.ElapsedMilliseconds - LastProgressMs < 5000) return;
            LastProgressMs = Watch.ElapsedMilliseconds;
            WriteJsonAtomic(System.IO.Path.Combine(Output, "progress.json"), Obj("status", "running", "started_utc", StartedUtc,
                "updated_utc", DateTime.UtcNow.ToString("o"), "elapsed_seconds", Watch.Elapsed.TotalSeconds, "current_path", current,
                "file_paths_seen", Progress.Files, "directory_rows_completed", DirectoryRows, "unique_file_identities", Seen.Count,
                "totals_so_far", Progress.Data()));
        }
    }
    public static string Run(string root, string outdir)
    {
        // PowerShell 5 hosts can otherwise retain .NET's legacy path parsing even on 4.8.
        // These process-local switches enable the explicit extended paths below; no OS policy is changed.
        AppContext.SetSwitch("Switch.System.IO.UseLegacyPathHandling", false);
        AppContext.SetSwitch("Switch.System.IO.BlockLongPaths", false);
        string fullRoot = Path.GetFullPath(root).TrimEnd('\\') + "\\";
        if (!String.Equals(fullRoot, @"C:\", StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("本工具仅允许审计 C:\\ 根目录。");
        string output = Path.GetFullPath(outdir).TrimEnd('\\');
        if (!output.StartsWith(@"C:\", StringComparison.OrdinalIgnoreCase) || output.Length <= 3) throw new ArgumentException("输出必须是 C: 上独立的新目录。");
        ValidateNoReparseAncestors(output);
        if (Directory.Exists(output) || File.Exists(output)) throw new IOException("输出目录必须不存在；不会覆盖旧审计：" + output);
        Directory.CreateDirectory(output);
        Scan scan = new Scan { Root = fullRoot, Output = output, StartedUtc = DateTime.UtcNow.ToString("o") };
        Stack<Frame> frames = new Stack<Frame>(); Totals finalTotals = null; Exception fatal = null;
        using (scan.Index = NewWriter(Path.Combine(output, "all-files-index-local-only.csv")))
        using (scan.DirectoriesCsv = NewWriter(Path.Combine(output, "directories.csv")))
        using (scan.DirectoriesJson = NewWriter(Path.Combine(output, "directories.json")))
        using (scan.Large = NewWriter(Path.Combine(output, "large-files.csv")))
        using (scan.Errors = NewWriter(Path.Combine(output, "errors.csv")))
        using (scan.Reparse = NewWriter(Path.Combine(output, "reparse-skipped.csv")))
        {
            const string fileHeader = "path,logical_bytes,allocation_bytes,hard_links,file_id,duplicate_identity,first_logical_assignment,first_allocation_assignment,created_utc,modified_utc,attributes,error";
            scan.Index.WriteLine(fileHeader); scan.Large.WriteLine(fileHeader);
            scan.Errors.WriteLine("observed_utc,path,operation,win32_code,message");
            scan.Reparse.WriteLine("path,attributes,reason");
            scan.DirectoriesCsv.WriteLine("path,depth,recursive_files,recursive_directories_including_self,recursive_logical_known_bytes,recursive_allocation_path_known_bytes,recursive_unique_logical_assigned_bytes,recursive_unique_allocation_assigned_bytes,unknown_logical_files,unknown_allocation_files,unknown_identity_files,duplicate_identity_paths,reparse_skipped,errors,output_excluded,direct_files,direct_logical_known_bytes,direct_allocation_path_known_bytes");
            scan.DirectoriesJson.Write("[\n");
            frames.Push(new Frame { Path = fullRoot, Depth = 0 }); scan.Progress.Directories = 1; scan.ProgressFile(fullRoot, true);
            try
            {
                while (frames.Count > 0)
                {
                    Frame frame = frames.Peek();
                    if (!frame.Started)
                    {
                        frame.Started = true;
                        try
                        {
                            FileAttributes currentAttributes = File.GetAttributes(Extended(frame.Path));
                            if ((currentAttributes & FileAttributes.ReparsePoint) != 0) SkipReparse(scan, frame, frame.Path, (uint)currentAttributes, "directory_became_reparse_before_enumeration");
                            else frame.Entries = Directory.EnumerateFileSystemEntries(Extended(frame.Path)).GetEnumerator();
                        }
                        catch (Exception ex) { scan.Error(frame, frame.Path, "enumerate_open", ex.Message, null); }
                    }
                    string entry = null; bool more = false;
                    if (frame.Entries != null)
                    {
                        try { more = frame.Entries.MoveNext(); if (more) entry = Normal(frame.Entries.Current); }
                        catch (Exception ex) { scan.Error(frame, frame.Path, "enumerate_next", ex.Message, null); }
                    }
                    if (!more)
                    {
                        if (frame.Entries != null) frame.Entries.Dispose();
                        frames.Pop(); EmitDirectory(scan, frame);
                        if (frames.Count > 0) frames.Peek().Recursive.Add(frame.Recursive); else finalTotals = frame.Recursive;
                        if (scan.Watch.ElapsedMilliseconds - scan.LastProgressMs >= 5000) scan.ProgressFile(frame.Path, true);
                        continue;
                    }
                    if (String.Equals(entry.TrimEnd('\\'), output, StringComparison.OrdinalIgnoreCase))
                    {
                        frame.Direct.OutputExcluded++; frame.Recursive.OutputExcluded++; scan.Progress.OutputExcluded++;
                        WriteCsv(scan.Reparse, new object[] { entry, null, "audit_output_excluded" }); continue;
                    }
                    FileAttributes attributes;
                    try { attributes = File.GetAttributes(Extended(entry)); }
                    catch (Exception ex) { scan.Error(frame, entry, "get_attributes", ex.Message, null); continue; }
                    if ((attributes & FileAttributes.ReparsePoint) != 0)
                    {
                        SkipReparse(scan, frame, entry, (uint)attributes, "reparse_attribute_no_follow"); continue;
                    }
                    if ((attributes & FileAttributes.Directory) != 0)
                    {
                        frame.Direct.Directories++; scan.Progress.Directories++; frames.Push(new Frame { Path = entry, Depth = frame.Depth + 1 }); continue;
                    }
                    FileRow row = ReadFile(scan, frame, entry, attributes);
                    if (row == null) continue;
                    Totals delta = CountFile(scan, row); frame.Direct.Add(delta); frame.Recursive.Add(delta); scan.Progress.Add(delta);
                    WriteCsv(scan.Index, row.Columns());
                    if (row.Logical.HasValue && row.Logical.Value >= LargeThreshold) { WriteCsv(scan.Large, row.Columns()); scan.LargeRows++; }
                    UpdateTop(scan, row); scan.ProgressFile(entry, false);
                }
            }
            catch (Exception ex) { fatal = ex; }
            finally
            {
                while (frames.Count > 0) { Frame frame = frames.Pop(); if (frame.Entries != null) frame.Entries.Dispose(); }
                scan.DirectoriesJson.Write("\n]\n");
            }
        }
        string status = fatal != null ? "incomplete_fatal_error" : scan.Progress.Errors > 0 ? "completed_with_access_or_metadata_gaps" : "completed_metadata_scan";
        WriteJsonAtomic(Path.Combine(output, "top100-files.json"), scan.Top.ConvertAll<object>(delegate(FileRow row) { return row.Data(); }));
        Dictionary<string, object> summary = Obj("schema_version", 1, "status", status, "root", fullRoot, "output", output,
            "started_utc", scan.StartedUtc, "finished_utc", DateTime.UtcNow.ToString("o"), "elapsed_seconds", scan.Watch.Elapsed.TotalSeconds,
            "is_administrator_required", false, "file_paths_seen", scan.Progress.Files, "directory_rows_completed", scan.DirectoryRows,
            "unique_file_identities", scan.Seen.Count, "large_threshold_bytes", LargeThreshold, "large_file_rows", scan.LargeRows,
            "totals", (finalTotals ?? scan.Progress).Data(), "fatal_error", fatal == null ? null : fatal.ToString(),
            "semantics", new[] {
                "Read-only metadata audit: no file-content reads, deletes, process command-line queries, or reparse traversal.",
                "Logical bytes count each observed path. Allocation path bytes also count hard-link aliases repeatedly; never use this column as unique disk usage.",
                "Unique totals use volume serial plus file ID. Each identity contributes known logical/allocation bytes once, at its first path with that metadata available.",
                "Per-directory unique assignment depends on native traversal order; it is NOT the bytes that deleting that directory would free. A retained hard link elsewhere keeps its allocation.",
                "AllocationSize is the filesystem-reported FILE_STANDARD_INFO value. Missing API results remain null on file rows and are counted as unknown; known totals are lower bounds when gaps exist.",
                "Only regular-file default-stream metadata is counted. Directory allocation, NTFS metadata, alternate data streams, snapshots, reserved storage and filesystem compression/dedup accounting may differ from volume used bytes.",
                "The live filesystem can change during this scan. Native directory enumeration order is not sorted, and the audit output directory is excluded.",
                "all-files-index-local-only.csv stays on the VM. Transfer/zip only directories.*, large-files.csv, top100-files.json, errors.csv, reparse-skipped.csv, summary.json and progress.json."
            });
        try
        {
            DriveInfo drive = new DriveInfo("C");
            summary["volume_at_finish"] = Obj("format", drive.DriveFormat, "total_bytes", drive.TotalSize, "total_free_bytes", drive.TotalFreeSpace,
                "available_free_bytes", drive.AvailableFreeSpace, "used_bytes", drive.TotalSize - drive.TotalFreeSpace);
        }
        catch (Exception ex) { summary["volume_query_error"] = ex.Message; }
        WriteJsonAtomic(Path.Combine(output, "summary.json"), summary);
        WriteJsonAtomic(Path.Combine(output, "progress.json"), Obj("status", status, "finished_utc", DateTime.UtcNow.ToString("o"), "file_paths_seen", scan.Progress.Files, "directory_rows_completed", scan.DirectoryRows, "summary", "summary.json"));
        if (fatal != null) throw new IOException("审计不完整，已保留结果和 summary.json。", fatal);
        return Path.Combine(output, "summary.json");
    }
    private static FileRow ReadFile(Scan scan, Frame frame, string path, FileAttributes attributes)
    {
        FileRow row = new FileRow { Path = path, Attributes = (uint)attributes };
        using (SafeFileHandle handle = CreateFile(Extended(path), 0, ShareAll, IntPtr.Zero, OpenExisting, OpenReparsePoint | BackupSemantics, IntPtr.Zero))
        {
            if (handle.IsInvalid)
            {
                int code = Marshal.GetLastWin32Error(); row.Error = "CreateFile:" + code;
                scan.Error(frame, path, "metadata_open_access0", new System.ComponentModel.Win32Exception(code).Message, code);
            }
            else
            {
                ByHandleInfo identity;
                if (GetFileInformationByHandle(handle, out identity))
                {
                    if ((identity.Attributes & (uint)FileAttributes.ReparsePoint) != 0) { SkipReparse(scan, frame, path, identity.Attributes, "became_reparse_before_open"); return null; }
                    if ((identity.Attributes & (uint)FileAttributes.Directory) != 0) { scan.Error(frame, path, "metadata_race", "File path became a directory during audit; skipped.", null); return null; }
                    FileKey key = new FileKey { Volume = identity.VolumeSerial, Id = ((ulong)identity.IndexHigh << 32) | identity.IndexLow };
                    row.Id = key.ToString(); row.Links = identity.Links;
                    row.Logical = unchecked((long)(((ulong)identity.SizeHigh << 32) | identity.SizeLow));
                    row.CreatedUtc = Time(identity.Created); row.ModifiedUtc = Time(identity.Written); row.Attributes = identity.Attributes;
                }
                else
                {
                    int code = Marshal.GetLastWin32Error(); row.Error = "FileIdentity:" + code;
                    scan.Error(frame, path, "GetFileInformationByHandle", new System.ComponentModel.Win32Exception(code).Message, code);
                }
                StandardInfo standard;
                if (GetFileInformationByHandleEx(handle, 1, out standard, (uint)Marshal.SizeOf(typeof(StandardInfo))))
                {
                    if (standard.Directory) { scan.Error(frame, path, "metadata_race", "File path became a directory during audit; skipped.", null); return null; }
                    if (standard.AllocationSize >= 0) row.Allocation = standard.AllocationSize;
                    if (standard.EndOfFile >= 0) row.Logical = standard.EndOfFile;
                    row.Links = standard.NumberOfLinks;
                }
                else
                {
                    int code = Marshal.GetLastWin32Error(); row.Error = (row.Error == null ? "" : row.Error + ";") + "StandardInfo:" + code;
                    scan.Error(frame, path, "GetFileInformationByHandleEx(FileStandardInfo)", new System.ComponentModel.Win32Exception(code).Message, code);
                }
            }
        }
        if (!row.Logical.HasValue)
        {
            try { row.Logical = new FileInfo(Extended(path)).Length; }
            catch (Exception ex) { scan.Error(frame, path, "fallback_length_metadata", ex.Message, null); }
        }
        return row;
    }
    private static Totals CountFile(Scan scan, FileRow row)
    {
        Totals t = new Totals { Files = 1 };
        if (row.Logical.HasValue) t.LogicalKnownBytes = row.Logical.Value; else t.UnknownLogicalFiles++;
        if (row.Allocation.HasValue) t.AllocationPathKnownBytes = row.Allocation.Value; else t.UnknownAllocationFiles++;
        if (row.Id == null) { t.UnknownIdentityFiles++; return t; }
        string[] pieces = row.Id.Split(':');
        FileKey key = new FileKey { Volume = UInt32.Parse(pieces[0], NumberStyles.HexNumber), Id = UInt64.Parse(pieces[1], NumberStyles.HexNumber) };
        byte flags; bool seen = scan.Seen.TryGetValue(key, out flags); row.Duplicate = seen; if (seen) t.DuplicateIdentityPaths++;
        if (row.Logical.HasValue && (flags & 1) == 0) { flags |= 1; row.FirstLogical = true; t.UniqueLogicalAssignedBytes = row.Logical.Value; }
        if (row.Allocation.HasValue && (flags & 2) == 0) { flags |= 2; row.FirstAllocation = true; t.UniqueAllocationAssignedBytes = row.Allocation.Value; }
        scan.Seen[key] = flags; return t;
    }
    private static void UpdateTop(Scan scan, FileRow row)
    {
        if (!row.Logical.HasValue) return;
        if (scan.Top.Count == 100 && row.Logical.Value <= scan.Top[99].Logical.Value) return;
        scan.Top.Add(row); scan.Top.Sort(delegate(FileRow a, FileRow b) { return b.Logical.Value.CompareTo(a.Logical.Value); });
        if (scan.Top.Count > 100) scan.Top.RemoveAt(100);
    }
    private static void EmitDirectory(Scan scan, Frame frame)
    {
        Totals t = frame.Recursive;
        WriteCsv(scan.DirectoriesCsv, new object[] { frame.Path, frame.Depth, t.Files, t.Directories, t.LogicalKnownBytes, t.AllocationPathKnownBytes,
            t.UniqueLogicalAssignedBytes, t.UniqueAllocationAssignedBytes, t.UnknownLogicalFiles, t.UnknownAllocationFiles, t.UnknownIdentityFiles,
            t.DuplicateIdentityPaths, t.ReparseSkipped, t.Errors, t.OutputExcluded, frame.Direct.Files, frame.Direct.LogicalKnownBytes, frame.Direct.AllocationPathKnownBytes });
        if (!scan.FirstDirectory) scan.DirectoriesJson.Write(",\n"); scan.FirstDirectory = false;
        scan.DirectoriesJson.Write(Json(Obj("path", frame.Path, "depth", frame.Depth, "recursive", t.Data(), "direct", frame.Direct.Data())));
        scan.DirectoryRows++;
    }
    private static void SkipReparse(Scan scan, Frame frame, string path, uint attributes, string reason)
    {
        WriteCsv(scan.Reparse, new object[] { path, attributes, reason }); frame.Direct.ReparseSkipped++; frame.Recursive.ReparseSkipped++; scan.Progress.ReparseSkipped++;
    }
    private static string Time(NativeFileTime time)
    {
        try { return DateTime.FromFileTimeUtc(unchecked((long)(((ulong)time.High << 32) | time.Low))).ToString("o"); } catch { return null; }
    }
    private static void ValidateNoReparseAncestors(string path)
    {
        string current = path;
        while (!String.IsNullOrEmpty(current))
        {
            if ((Directory.Exists(current) || File.Exists(current)) && (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0) throw new IOException("输出路径不能经过重解析点：" + current);
            current = Path.GetDirectoryName(current);
        }
    }
    private static string Extended(string path) { return path.StartsWith(@"\\?\", StringComparison.Ordinal) ? path : @"\\?\" + path; }
    private static string Normal(string path) { return path.StartsWith(@"\\?\", StringComparison.Ordinal) ? path.Substring(4) : path; }
    private static StreamWriter NewWriter(string path)
    { return new StreamWriter(new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 65536), new UTF8Encoding(true), 65536); }
    private static void WriteCsv(StreamWriter writer, object[] values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (i > 0) writer.Write(','); writer.Write('"'); writer.Write(Convert.ToString(values[i], CultureInfo.InvariantCulture).Replace("\"", "\"\"")); writer.Write('"');
        }
        writer.WriteLine();
    }
    private static Dictionary<string, object> Obj(params object[] items)
    { Dictionary<string, object> d = new Dictionary<string, object>(); for (int i = 0; i < items.Length; i += 2) d[(string)items[i]] = items[i + 1]; return d; }
    private static void WriteJsonAtomic(string path, object value)
    {
        string temp = path + ".writing"; File.WriteAllText(temp, Json(value) + Environment.NewLine, new UTF8Encoding(false));
        if (File.Exists(path)) File.Replace(temp, path, null); else File.Move(temp, path);
    }
    private static string Json(object value) { StringBuilder b = new StringBuilder(); WriteJson(b, value); return b.ToString(); }
    private static void WriteJson(StringBuilder b, object value)
    {
        if (value == null) { b.Append("null"); return; }
        string s = value as string;
        if (s != null)
        {
            b.Append('"'); foreach (char c in s)
            {
                if (c == '"') b.Append("\\\""); else if (c == '\\') b.Append("\\\\"); else if (c == '\r') b.Append("\\r");
                else if (c == '\n') b.Append("\\n"); else if (c == '\t') b.Append("\\t"); else if (c < 32) b.Append("\\u" + ((int)c).ToString("x4")); else b.Append(c);
            }
            b.Append('"'); return;
        }
        if (value is bool) { b.Append((bool)value ? "true" : "false"); return; }
        IDictionary dictionary = value as IDictionary;
        if (dictionary != null)
        {
            b.Append('{'); bool first = true;
            foreach (DictionaryEntry item in dictionary) { if (!first) b.Append(','); first = false; WriteJson(b, Convert.ToString(item.Key)); b.Append(':'); WriteJson(b, item.Value); }
            b.Append('}'); return;
        }
        IEnumerable sequence = value as IEnumerable;
        if (sequence != null)
        { b.Append('['); bool first = true; foreach (object item in sequence) { if (!first) b.Append(','); first = false; WriteJson(b, item); } b.Append(']'); return; }
        b.Append(Convert.ToString(value, CultureInfo.InvariantCulture));
    }
}
