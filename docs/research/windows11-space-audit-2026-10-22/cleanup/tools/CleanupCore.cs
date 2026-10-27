// C# 5 / .NET Framework 4.8. No work is performed by loading this file.
// Add-Type references: System.dll, System.Core.dll, System.Runtime.Serialization.dll,
// System.IO.Compression.dll. Every operation is synchronous and fail-closed.
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace VmCleanup
{
    [DataContract]
    public sealed class FileRecord
    {
        [DataMember] public string Root { get; set; }
        [DataMember] public string RelativePath { get; set; }
        [DataMember] public string FullPath { get; set; }
        [DataMember] public long Length { get; set; }
        [DataMember] public long LastWriteUtcTicks { get; set; }
        [DataMember] public string Sha256 { get; set; }
        [DataMember] public uint VolumeSerialNumber { get; set; }
        [DataMember] public uint FileIndexHigh { get; set; }
        [DataMember] public uint FileIndexLow { get; set; }
        [DataMember] public string ArchiveEntry { get; set; }
    }
    [DataContract]
    public sealed class ArchiveManifest
    {
        [DataMember] public int SchemaVersion { get; set; }
        [DataMember] public string CreatedUtc { get; set; }
        [DataMember] public FileRecord[] Files { get; set; }
        [DataMember] public long TotalBytes { get; set; }
    }
    public sealed class DeleteResult
    {
        public string Root { get; set; }
        public string[] DeletedFiles { get; set; }
        public string[] DeletedDirectories { get; set; }
        public long DeletedLogicalBytes { get; set; }
        public bool RootStillExists { get; set; }
    }
    public sealed class RetainedScanResult
    {
        public FileRecord[] Files { get; set; }
        public string[] ReparsePoints { get; set; }
    }

    public static class CleanupCore
    {
        const uint GENERIC_READ = 0x80000000, GENERIC_WRITE = 0x40000000;
        const uint DELETE_ACCESS = 0x00010000, READ_ATTRIBUTES = 0x80;
        const uint SHARE_READ = 1, SHARE_WRITE = 2;
        const uint OPEN_EXISTING = 3, CREATE_NEW = 1;
        const uint BACKUP_SEMANTICS = 0x02000000, OPEN_REPARSE_POINT = 0x00200000;
        const uint DIRECTORY_ATTRIBUTE = 0x10, REPARSE_ATTRIBUTE = 0x400;
        const string ManifestName = "manifest.json";
        [StructLayout(LayoutKind.Sequential)] struct WinTime { public uint Low, High; }
        [StructLayout(LayoutKind.Sequential)] struct HandleInfo
        {
            public uint Attributes; public WinTime Creation, Access, Write;
            public uint Volume, SizeHigh, SizeLow, Links, IndexHigh, IndexLow;
        }
        [StructLayout(LayoutKind.Sequential, Pack=1)] struct Disposition
        { [MarshalAs(UnmanagedType.U1)] public bool DeleteFile; }
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
        static extern SafeFileHandle CreateFile(string path, uint access, uint share, IntPtr security,
            uint creation, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool GetFileInformationByHandle(SafeFileHandle handle, out HandleInfo info);
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
        static extern uint GetFinalPathNameByHandle(SafeFileHandle handle, StringBuilder buffer, uint size, uint flags);
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] static extern uint GetDriveType(string root);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool SetFileInformationByHandle(SafeFileHandle handle, int infoClass,
            ref Disposition info, uint size);

        static Exception WinError(string action, string path)
        { return new IOException(action + ": " + path, new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error())); }
        static string Normalize(string path)
        {
            if (String.IsNullOrWhiteSpace(path) || path.Length < 3 || !Char.IsLetter(path[0]) ||
                path[1] != ':' || path[2] != '\\' || path.StartsWith("\\", StringComparison.Ordinal))
                throw new ArgumentException("Only absolute local drive paths are allowed: " + path);
            if (path.IndexOf(':', 2) >= 0 || path.IndexOf('/') >= 0)
                throw new ArgumentException("Alternate streams and slash aliases are not allowed: " + path);
            foreach (string s in path.Substring(3).Split('\\'))
                if (s == "." || s == ".." || s.EndsWith(".", StringComparison.Ordinal) || s.EndsWith(" ", StringComparison.Ordinal))
                    throw new ArgumentException("Ambiguous path component: " + path);
            string full = Path.GetFullPath(path);
            uint kind = GetDriveType(Path.GetPathRoot(full));
            if (kind != 2 && kind != 3 && kind != 6)
                throw new IOException("Network, shared or unavailable drive rejected: " + path);
            return full.Length > 3 ? full.TrimEnd('\\') : full;
        }
        static string RootPath(string root)
        {
            root = Normalize(root);
            if (root.Length == 3) throw new ArgumentException("A drive root is not an allowed cleanup scope.");
            return root;
        }
        static bool EqualPath(string a, string b) { return String.Equals(a, b, StringComparison.OrdinalIgnoreCase); }
        static bool Within(string root, string path)
        { return EqualPath(root, path) || path.StartsWith(root + "\\", StringComparison.OrdinalIgnoreCase); }
        static HandleInfo Inspect(SafeFileHandle h, string expected, bool directory)
        {
            HandleInfo i;
            if (!GetFileInformationByHandle(h, out i)) throw WinError("Inspect handle", expected);
            if ((i.Attributes & REPARSE_ATTRIBUTE) != 0) throw new IOException("Reparse point rejected: " + expected);
            if (((i.Attributes & DIRECTORY_ATTRIBUTE) != 0) != directory) throw new IOException("Object type changed: " + expected);
            StringBuilder b = new StringBuilder(32768);
            uint n = GetFinalPathNameByHandle(h, b, (uint)b.Capacity, 0);
            if (n == 0 || n >= b.Capacity) throw WinError("Resolve handle path", expected);
            string actual = b.ToString();
            if (actual.StartsWith("\\\\?\\", StringComparison.Ordinal)) actual = actual.Substring(4);
            if (!EqualPath(Normalize(actual), expected)) throw new IOException("Resolved path mismatch: " + expected);
            return i;
        }
        static SafeFileHandle Open(string path, bool directory, uint access, uint share, uint creation)
        {
            SafeFileHandle h = CreateFile(path, access, share, IntPtr.Zero, creation,
                OPEN_REPARSE_POINT | (directory ? BACKUP_SEMANTICS : 0), IntPtr.Zero);
            if (h.IsInvalid) { Exception error=WinError("Open guarded path", path); h.Dispose(); throw error; }
            try { Inspect(h, path, directory); return h; } catch { h.Dispose(); throw; }
        }
        sealed class ParentLocks : IDisposable
        {
            readonly List<SafeFileHandle> handles = new List<SafeFileHandle>();
            public ParentLocks(string path)
            {
                try
                {
                    string parent = Path.GetDirectoryName(path);
                    List<string> parents = new List<string>();
                    while (!String.IsNullOrEmpty(parent))
                    { parents.Add(parent); parent = Path.GetDirectoryName(parent); }
                    parents.Reverse();
                    foreach (string p in parents)
                        handles.Add(Open(Normalize(p), true, READ_ATTRIBUTES, SHARE_READ | SHARE_WRITE, OPEN_EXISTING));
                }
                catch { Dispose(); throw; }
            }
            public void Dispose() { for (int i=handles.Count-1; i>=0; --i) handles[i].Dispose(); handles.Clear(); }
        }
        static FileRecord Record(string root, string path, HandleInfo i)
        {
            long ft = ((long)i.Write.High << 32) | i.Write.Low;
            return new FileRecord { Root=root, FullPath=path,
                RelativePath=EqualPath(root,path)?Path.GetFileName(path):path.Substring(root.Length+1),
                Length=((long)i.SizeHigh<<32)|i.SizeLow, LastWriteUtcTicks=DateTime.FromFileTimeUtc(ft).Ticks,
                VolumeSerialNumber=i.Volume, FileIndexHigh=i.IndexHigh, FileIndexLow=i.IndexLow };
        }
        static FileStream Stream(SafeFileHandle handle, FileAccess access)
        {
            // The wrapper does not own the underlying handle; deletion uses the same still-open handle.
            return new FileStream(new SafeFileHandle(handle.DangerousGetHandle(), false), access, 1024*1024, false);
        }
        static string Hex(byte[] bytes) { return BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant(); }
        static string HashStream(System.IO.Stream stream)
        { using (SHA256 sha=SHA256.Create()) return Hex(sha.ComputeHash(stream)); }
        static void RequireHash(FileRecord f)
        {
            if (f == null || String.IsNullOrEmpty(f.Sha256) || f.Sha256.Length != 64 ||
                f.Sha256.Any(c => !Uri.IsHexDigit(c))) throw new ArgumentException("A reviewed SHA256 is required.");
        }
        static void Match(FileRecord expected, HandleInfo i, string root, string path)
        {
            RequireHash(expected);
            FileRecord now=Record(root,path,i);
            if (now.Length != expected.Length || now.LastWriteUtcTicks != expected.LastWriteUtcTicks ||
                now.VolumeSerialNumber != expected.VolumeSerialNumber || now.FileIndexHigh != expected.FileIndexHigh ||
                now.FileIndexLow != expected.FileIndexLow) throw new IOException("Reviewed file identity changed: " + path);
        }
        static void ListDirectory(string root, string current, List<FileRecord> result)
        {
            string[] children;
            using (ParentLocks locks=new ParentLocks(current))
            using (SafeFileHandle d=Open(current,true,READ_ATTRIBUTES,SHARE_READ|SHARE_WRITE,OPEN_EXISTING))
                children=Directory.GetFileSystemEntries(current);
            Array.Sort(children, StringComparer.OrdinalIgnoreCase);
            foreach (string raw in children)
            {
                string path=Normalize(raw);
                using (ParentLocks locks=new ParentLocks(path))
                {
                    FileAttributes a=File.GetAttributes(path);
                    if ((a & FileAttributes.ReparsePoint)!=0) throw new IOException("Reparse point in scope: " + path);
                    if ((a & FileAttributes.Directory)!=0) ListDirectory(root,path,result);
                    else using (SafeFileHandle h=Open(path,false,GENERIC_READ,SHARE_READ,OPEN_EXISTING))
                        result.Add(Record(root,path,Inspect(h,path,false)));
                }
            }
        }
        public static FileRecord[] ListFiles(string root)
        {
            root=RootPath(root);
            List<FileRecord> files=new List<FileRecord>();
            using (ParentLocks locks=new ParentLocks(root))
            {
                FileAttributes a=File.GetAttributes(root);
                if ((a & FileAttributes.ReparsePoint)!=0) throw new IOException("Reparse root rejected: " + root);
                if ((a & FileAttributes.Directory)!=0) ListDirectory(root,root,files);
                else using (SafeFileHandle h=Open(root,false,GENERIC_READ,SHARE_READ,OPEN_EXISTING))
                    files.Add(Record(root,root,Inspect(h,root,false)));
            }
            return files.ToArray();
        }
        static bool Excluded(string path, string[] excludedRoots)
        {
            foreach (string excluded in excludedRoots)
                if (Within(excluded,path)) return true;
            return false;
        }
        static void ListRetainedPath(string root, string path, string[] excludedRoots,
            List<FileRecord> files, List<string> reparsePoints)
        {
            // Check before enumerating or opening an excluded subtree, including its root.
            if (Excluded(path,excludedRoots)) return;
            using (ParentLocks locks=new ParentLocks(path))
            {
                FileAttributes attributes=File.GetAttributes(path);
                if ((attributes & FileAttributes.ReparsePoint)!=0)
                { reparsePoints.Add(path); return; }
                if ((attributes & FileAttributes.Directory)==0)
                {
                    using (SafeFileHandle h=Open(path,false,GENERIC_READ,SHARE_READ,OPEN_EXISTING))
                        files.Add(Record(root,path,Inspect(h,path,false)));
                    return;
                }
                using (SafeFileHandle h=Open(path,true,READ_ATTRIBUTES,SHARE_READ|SHARE_WRITE,OPEN_EXISTING))
                {
                    string[] children=Directory.GetFileSystemEntries(path);
                    Array.Sort(children,StringComparer.OrdinalIgnoreCase);
                    foreach (string child in children)
                    {
                        string normalized=Normalize(child);
                        if (!Within(root,normalized)) throw new IOException("Retained scan escaped root: " + normalized);
                        ListRetainedPath(root,normalized,excludedRoots,files,reparsePoints);
                    }
                }
            }
        }
        public static RetainedScanResult ListRetainedFiles(string root, string[] excludedRoots)
        {
            root=RootPath(root);
            if (excludedRoots==null) throw new ArgumentNullException("excludedRoots");
            string[] excluded=excludedRoots.Select(RootPath).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
            List<FileRecord> files=new List<FileRecord>();
            List<string> reparsePoints=new List<string>();
            // This separate read-only scanner deliberately reports links without traversing
            // them. Strict ListFiles and all delete/archival paths retain their rejection rules.
            ListRetainedPath(root,root,excluded,files,reparsePoints);
            return new RetainedScanResult { Files=files.ToArray(), ReparsePoints=reparsePoints.ToArray() };
        }
        public static string Hash(string path)
        {
            path=Normalize(path);
            using (ParentLocks locks=new ParentLocks(path))
            using (SafeFileHandle h=Open(path,false,GENERIC_READ,SHARE_READ,OPEN_EXISTING))
            using (FileStream s=Stream(h,FileAccess.Read)) return HashStream(s);
        }
        public static void HashRecords(FileRecord[] records)
        {
            if (records==null) throw new ArgumentNullException("records");
            string[] paths=new string[records.Length];
            for (int i=0; i<records.Length; ++i)
            {
                if (records[i]==null) throw new ArgumentException("Null file record.");
                paths[i]=Normalize(records[i].FullPath);
            }
            string[] hashes=new string[records.Length];
            Parallel.For(0,records.Length,new ParallelOptions { MaxDegreeOfParallelism=4 },
                i => { hashes[i]=Hash(paths[i]); });
            // Parallel.For waits for started workers and throws on any failure. Commit hashes
            // to caller records only when the entire batch succeeded; never retry internally.
            for (int i=0; i<records.Length; ++i) records[i].Sha256=hashes[i];
        }
        static string EntryName(string path)
        { return Char.ToUpperInvariant(path[0]) + "/" + path.Substring(3).Replace('\\','/'); }
        static FileRecord Clone(FileRecord f)
        {
            return new FileRecord { Root=f.Root, RelativePath=f.RelativePath, FullPath=f.FullPath,
                Length=f.Length, LastWriteUtcTicks=f.LastWriteUtcTicks, Sha256=f.Sha256.ToLowerInvariant(),
                VolumeSerialNumber=f.VolumeSerialNumber, FileIndexHigh=f.FileIndexHigh, FileIndexLow=f.FileIndexLow,
                ArchiveEntry=EntryName(f.FullPath) };
        }
        static FileRecord[] Reviewed(FileRecord[] files, string scope)
        {
            if (files==null) throw new ArgumentNullException("files");
            HashSet<string> names=new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            List<FileRecord> list=new List<FileRecord>();
            foreach (FileRecord f in files)
            {
                RequireHash(f); string root=RootPath(f.Root), path=Normalize(f.FullPath);
                if (!EqualPath(f.Root,root) || !EqualPath(f.FullPath,path) || !Within(root,path) ||
                    (scope!=null && (!EqualPath(root,scope) || !Within(scope,path))))
                    throw new ArgumentException("Record outside reviewed root: " + path);
                string rel=EqualPath(root,path)?Path.GetFileName(path):path.Substring(root.Length+1);
                if (!String.Equals(rel,f.RelativePath,StringComparison.OrdinalIgnoreCase))
                    throw new ArgumentException("Relative path mismatch: " + path);
                if (!names.Add(path)) throw new ArgumentException("Duplicate reviewed file: " + path);
                list.Add(Clone(f));
            }
            return list.OrderBy(x=>x.FullPath,StringComparer.OrdinalIgnoreCase).ToArray();
        }
        static string CopyAndHash(System.IO.Stream source, System.IO.Stream destination)
        {
            byte[] buffer=new byte[1024*1024];
            using (SHA256 sha=SHA256.Create())
            {
                int n; while ((n=source.Read(buffer,0,buffer.Length))>0)
                { destination.Write(buffer,0,n); sha.TransformBlock(buffer,0,n,buffer,0); }
                sha.TransformFinalBlock(new byte[0],0,0); return Hex(sha.Hash);
            }
        }
        public static ArchiveManifest ArchiveSelected(FileRecord[] files, string zipPath)
        {
            FileRecord[] selected=Reviewed(files,null); zipPath=Normalize(zipPath);
            foreach (FileRecord f in selected)
                if (Within(f.Root,zipPath)) throw new ArgumentException("Archive must be outside every source root.");
            ArchiveManifest manifest=new ArchiveManifest { SchemaVersion=1, CreatedUtc=DateTime.UtcNow.ToString("o"),
                Files=selected, TotalBytes=selected.Sum(x=>x.Length) };
            using (ParentLocks zipLocks=new ParentLocks(zipPath))
            using (SafeFileHandle zh=Open(zipPath,false,GENERIC_READ|GENERIC_WRITE,0,CREATE_NEW))
            using (FileStream output=Stream(zh,FileAccess.ReadWrite))
            {
                using (ZipArchive zip=new ZipArchive(output,ZipArchiveMode.Create,true,Encoding.UTF8))
                {
                    foreach (FileRecord f in selected)
                    {
                        using (ParentLocks locks=new ParentLocks(f.FullPath))
                        using (SafeFileHandle h=Open(f.FullPath,false,GENERIC_READ,SHARE_READ,OPEN_EXISTING))
                        {
                            Match(f,Inspect(h,f.FullPath,false),f.Root,f.FullPath);
                            ZipArchiveEntry e=zip.CreateEntry(f.ArchiveEntry,CompressionLevel.Optimal);
                            using (FileStream input=Stream(h,FileAccess.Read))
                            using (System.IO.Stream dest=e.Open())
                                if (!String.Equals(CopyAndHash(input,dest),f.Sha256,StringComparison.OrdinalIgnoreCase))
                                    throw new IOException("Source hash changed while archiving: " + f.FullPath);
                            Match(f,Inspect(h,f.FullPath,false),f.Root,f.FullPath);
                        }
                    }
                    ZipArchiveEntry me=zip.CreateEntry(ManifestName,CompressionLevel.Optimal);
                    using (System.IO.Stream ms=me.Open()) new DataContractJsonSerializer(typeof(ArchiveManifest)).WriteObject(ms,manifest);
                }
                output.Flush(true);
            }
            // A failed archive is deliberately left for diagnosis; never delete source files here.
            return VerifyArchive(zipPath);
        }
        public static ArchiveManifest VerifyArchive(string zipPath)
        {
            zipPath=Normalize(zipPath);
            using (ParentLocks locks=new ParentLocks(zipPath))
            using (SafeFileHandle h=Open(zipPath,false,GENERIC_READ,SHARE_READ,OPEN_EXISTING))
            using (FileStream input=Stream(h,FileAccess.Read))
            using (ZipArchive zip=new ZipArchive(input,ZipArchiveMode.Read,false,Encoding.UTF8))
            {
                Dictionary<string,ZipArchiveEntry> entries=new Dictionary<string,ZipArchiveEntry>(StringComparer.OrdinalIgnoreCase);
                foreach (ZipArchiveEntry e in zip.Entries)
                    if (entries.ContainsKey(e.FullName)) throw new IOException("Duplicate ZIP entry: " + e.FullName);
                    else entries.Add(e.FullName,e);
                ZipArchiveEntry me;
                if (!entries.TryGetValue(ManifestName,out me) || me.Length>128L*1024*1024)
                    throw new IOException("Missing or oversized archive manifest.");
                ArchiveManifest m;
                using (System.IO.Stream ms=me.Open()) m=(ArchiveManifest)new DataContractJsonSerializer(typeof(ArchiveManifest)).ReadObject(ms);
                if (m==null || m.SchemaVersion!=1 || m.Files==null) throw new IOException("Unsupported archive manifest.");
                FileRecord[] reviewed=Reviewed(m.Files,null);
                if (entries.Count!=reviewed.Length+1 || m.TotalBytes!=reviewed.Sum(x=>x.Length))
                    throw new IOException("Archive file count or total length mismatch.");
                Dictionary<string,FileRecord> originals=m.Files.ToDictionary(x=>x.FullPath,StringComparer.OrdinalIgnoreCase);
                foreach (FileRecord f in reviewed)
                {
                    FileRecord original=originals[f.FullPath];
                    if (!String.Equals(original.ArchiveEntry,f.ArchiveEntry,StringComparison.Ordinal))
                        throw new IOException("Unsafe archive mapping: " + original.ArchiveEntry);
                    ZipArchiveEntry e;
                    if (!entries.TryGetValue(f.ArchiveEntry,out e) || e.Length!=f.Length)
                        throw new IOException("Missing or wrong-sized archive entry: " + f.ArchiveEntry);
                    using (System.IO.Stream es=e.Open())
                        if (!String.Equals(HashStream(es),f.Sha256,StringComparison.OrdinalIgnoreCase))
                            throw new IOException("Archive SHA256 mismatch: " + f.ArchiveEntry);
                }
                return m;
            }
        }
        static void ValidateReviewedFile(FileRecord f)
        {
            using (ParentLocks locks=new ParentLocks(f.FullPath))
            using (SafeFileHandle h=Open(f.FullPath,false,GENERIC_READ,SHARE_READ,OPEN_EXISTING))
            {
                Match(f,Inspect(h,f.FullPath,false),f.Root,f.FullPath);
                using (FileStream s=Stream(h,FileAccess.Read))
                    if (!String.Equals(HashStream(s),f.Sha256,StringComparison.OrdinalIgnoreCase))
                        throw new IOException("Reviewed hash mismatch: " + f.FullPath);
            }
        }
        public static void ValidateRecords(FileRecord[] records)
        {
            if (records==null) throw new ArgumentNullException("records");
            Parallel.ForEach(records,new ParallelOptions { MaxDegreeOfParallelism=4 },
                f => { ValidateReviewedFile(f); });
        }
        static void MarkDelete(SafeFileHandle handle, string path)
        {
            Disposition d=new Disposition { DeleteFile=true };
            if (!SetFileInformationByHandle(handle,4,ref d,(uint)Marshal.SizeOf(typeof(Disposition))))
                throw WinError("Delete by verified handle",path);
        }
        static bool SameObject(HandleInfo a, HandleInfo b)
        { return a.Volume==b.Volume && a.IndexHigh==b.IndexHigh && a.IndexLow==b.IndexLow; }
        static void RemoveEmptyDirectories(string root, string path, List<string> removed, HandleInfo? expected)
        {
            using (ParentLocks locks=new ParentLocks(path))
            {
                HandleInfo original;
                // Keep this directory immovable while descending, without DELETE access: child
                // ParentLocks deliberately do not share DELETE with other directory handles.
                using (SafeFileHandle h=Open(path,true,READ_ATTRIBUTES,SHARE_READ|SHARE_WRITE,OPEN_EXISTING))
                {
                    original=Inspect(h,path,true);
                    if (expected.HasValue && !SameObject(original,expected.Value))
                        throw new IOException("Reviewed directory was replaced: " + path);
                    foreach (string child in Directory.GetDirectories(path))
                    {
                        string normalized=Normalize(child);
                        if (!Within(root,normalized)) throw new IOException("Directory escaped scope.");
                        RemoveEmptyDirectories(root,normalized,removed,null);
                    }
                }
                using (SafeFileHandle h=Open(path,true,READ_ATTRIBUTES|DELETE_ACCESS,SHARE_READ|SHARE_WRITE,OPEN_EXISTING))
                {
                    if (!SameObject(original,Inspect(h,path,true)))
                        throw new IOException("Directory replaced during empty-directory cleanup: " + path);
                    // New files are preserved. A new directory reparse is refused above; a new
                    // entry racing this check makes the kernel reject deletion as non-empty.
                    if (Directory.GetFileSystemEntries(path).Length==0)
                    { MarkDelete(h,path); removed.Add(path); }
                }
            }
        }
        public static DeleteResult DeleteReviewed(string root, FileRecord[] expected)
        {
            root=RootPath(root); FileRecord[] reviewed=Reviewed(expected,root);
            if (reviewed.Length==0) throw new ArgumentException("An empty file list cannot authorize directory deletion.");
            List<string> deleted=new List<string>(), dirs=new List<string>(); long bytes=0;
            bool isDirectory; HandleInfo? rootIdentity=null;
            using (ParentLocks locks=new ParentLocks(root))
            {
                FileAttributes a=File.GetAttributes(root);
                if ((a & FileAttributes.ReparsePoint)!=0) throw new IOException("Reparse root rejected.");
                isDirectory=(a & FileAttributes.Directory)!=0;
                SafeFileHandle rootLock=isDirectory?Open(root,true,READ_ATTRIBUTES,SHARE_READ|SHARE_WRITE,OPEN_EXISTING):null;
                try
                {
                    if (rootLock!=null) rootIdentity=Inspect(rootLock,root,true);
                    FileRecord[] current=ListFiles(root);
                    HashSet<string> wanted=new HashSet<string>(reviewed.Select(f=>f.FullPath),StringComparer.OrdinalIgnoreCase);
                    if (current.Length!=reviewed.Length || current.Any(f=>!wanted.Contains(f.FullPath)))
                        throw new IOException("File set differs from reviewed manifest; no files deleted: " + root);
                    // All files and their hashes are checked before the first deletion. Subsequent
                    // races fail per file and may leave a partially completed deletion; never retry.
                    ValidateRecords(reviewed);
                    foreach (FileRecord f in reviewed)
                    {
                        using (ParentLocks fileLocks=new ParentLocks(f.FullPath))
                        using (SafeFileHandle h=Open(f.FullPath,false,GENERIC_READ|DELETE_ACCESS,SHARE_READ,OPEN_EXISTING))
                        {
                            Match(f,Inspect(h,f.FullPath,false),f.Root,f.FullPath);
                            using (FileStream s=Stream(h,FileAccess.Read))
                                if (!String.Equals(HashStream(s),f.Sha256,StringComparison.OrdinalIgnoreCase))
                                    throw new IOException("Reviewed hash changed at deletion: " + f.FullPath);
                            MarkDelete(h,f.FullPath);
                        }
                        deleted.Add(f.FullPath); bytes+=f.Length;
                    }
                }
                finally { if (rootLock!=null) rootLock.Dispose(); }
                if (isDirectory) RemoveEmptyDirectories(root,root,dirs,rootIdentity);
            }
            return new DeleteResult { Root=root, DeletedFiles=deleted.ToArray(), DeletedDirectories=dirs.ToArray(),
                DeletedLogicalBytes=bytes, RootStillExists=File.Exists(root)||Directory.Exists(root) };
        }
    }
}
