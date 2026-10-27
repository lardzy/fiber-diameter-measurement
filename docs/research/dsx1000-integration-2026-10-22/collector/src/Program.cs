using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DsxCollector
{
    internal static class Program
    {
        public const string Version = "1.0.0";
        [STAThread]
        private static int Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            if (args.Length >= 2 && args[0] == "--self-test")
            {
                Directory.CreateDirectory(args[1]);
                var results = new List<object>();
                try { results.AddRange(CoreSelfTests.Run(Path.Combine(args[1], "core"))); }
                catch (Exception ex) { results.Add(new { name = "core_unhandled", passed = false, detail = ex.ToString() }); }
                try { results.AddRange(RdkSelfTests.Run(Path.Combine(args[1], "rdk"))); }
                catch (Exception ex) { results.Add(new { name = "rdk_unhandled", passed = false, detail = ex.ToString() }); }
                string json = Util.Json(results);
                File.WriteAllText(Path.Combine(args[1], "self-tests.json"), json, new System.Text.UTF8Encoding(false));
                return json.IndexOf("\"passed\":false", StringComparison.OrdinalIgnoreCase) >= 0 ? 1 : 0;
            }
            if (args.Length >= 2 && args[0] == "--ui-smoke")
            {
                Directory.CreateDirectory(args[1]);
                using (var form = new CollectorForm())
                {
                    form.Show(); Application.DoEvents();
                    form.SaveUiSmoke(args[1]);
                    form.Close();
                }
                return 0;
            }
            Application.ThreadException += delegate(object sender, System.Threading.ThreadExceptionEventArgs e)
            { MessageBox.Show(Util.Redact(e.Exception.Message), "工具错误：已写入记录保留", MessageBoxButtons.OK, MessageBoxIcon.Error); };
            Application.Run(new CollectorForm());
            return 0;
        }
    }

    internal sealed class CollectorForm : Form
    {
        private EvidenceSession session;
        private ObservationRecorder recorder;
        private RdkClient rdk;
        private bool working, observing, identityMatched, remote, capturePending, sealedSession, collectionFault;
        private string currentCapture, lastShotState, inventoryRoot;
        private string lastRecorderError;
        private readonly Timer healthTimer = new Timer();
        private decimal? reportedX, reportedY, reportedZ;
        private readonly TabControl tabs = new TabControl();
        private readonly Label status = new Label();
        private readonly TextBox activity = new TextBox();
        private readonly TextBox sessionRoot = new TextBox(), installRoot = new TextBox(), imageRoot = new TextBox(), logRoot = new TextBox();
        private readonly TextBox host = new TextBox(), shotRoot = new TextBox();
        private readonly TextBox xmin = new TextBox(), xmax = new TextBox(), ymin = new TextBox(), ymax = new TextBox(), zmin = new TextBox(), zmax = new TextBox();
        private readonly TextBox targetX = new TextBox(), targetY = new TextBox(), deltaZ = new TextBox();
        private readonly TextBox reported = new TextBox();
        private readonly NumericUpDown port = new NumericUpDown(), deadline = new NumericUpDown();
        private readonly ComboBox speed = new ComboBox(), template = new ComboBox();
        private readonly CheckBox optIn = new CheckBox(), normalStart = new CheckBox(), stopReady = new CheckBox(), initReady = new CheckBox(), ordinarySnap = new CheckBox();
        private readonly CheckedListBox samples = new CheckedListBox();
        private readonly Dictionary<string, TextBox> fields = new Dictionary<string, TextBox>();
        private readonly List<Button> rdkActions = new List<Button>();
        private Button createSession, inventoryButton, startObserve, stopObserve, exportButton, copyButton, connectButton, disconnectButton, finishCapture;
        private Button takeControl, releaseControl, moveXY, moveZ, takeShot, stopShot, positionButton, errorButton, shotStatus;

        public CollectorForm()
        {
            Text = "DSX1000 现场证据收集器 " + Program.Version;
            Font = new Font("Microsoft YaHei UI", 9F);
            StartPosition = FormStartPosition.CenterScreen;
            Size = new Size(1160, 820); MinimumSize = new Size(920, 680);
            AutoScaleDimensions = new SizeF(96F, 96F);
            AutoScaleMode = AutoScaleMode.Dpi;
            var shell = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 4, ColumnCount = 1, Padding = new Padding(10) };
            shell.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
            shell.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            shell.RowStyles.Add(new RowStyle(SizeType.Absolute, 104));
            shell.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
            shell.Controls.Add(new Label { Text = "先记录原厂操作，再逐项验证接口", Dock = DockStyle.Fill, Font = new Font(Font.FontFamily, 15, FontStyle.Bold), TextAlign = ContentAlignment.MiddleLeft }, 0, 0);
            tabs.Dock = DockStyle.Fill; shell.Controls.Add(tabs, 0, 1);
            activity.Multiline = true; activity.ReadOnly = true; activity.ScrollBars = ScrollBars.Vertical; activity.Dock = DockStyle.Fill;
            shell.Controls.Add(activity, 0, 2);
            status.Text = "默认观察模式：未建立 RDK 连接。"; status.Dock = DockStyle.Fill; status.TextAlign = ContentAlignment.MiddleLeft;
            shell.Controls.Add(status, 0, 3); Controls.Add(shell);
            BuildObservePage(); BuildControlPage(); BuildHelpPage();
            sessionRoot.Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "DSXEvidenceCollector", "Sessions");
            installRoot.Text = Inventory.DiscoverRoot() ?? "";
            inventoryRoot = installRoot.Text;
            logRoot.Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonDocuments), "..", "OLYMPUS", "DSX-BSW", "Log", "CurrentLog");
            logRoot.Text = Path.GetFullPath(logRoot.Text);
            port.Minimum = 1; port.Maximum = 65535; port.Value = 35001;
            deadline.Minimum = 2; deadline.Maximum = 600; deadline.Value = 30;
            shotRoot.Text = @"C:\Users\Public\DSXEvidenceShots";
            installRoot.TextChanged += delegate { identityMatched = false; UpdateGates(); };
            foreach (CheckBox check in new[] { optIn, normalStart, stopReady, initReady, ordinarySnap }) check.CheckedChanged += delegate { UpdateGates(); };
            FormClosing += OnClosing;
            Load += delegate { MinimumSize = new Size(920, 680); Size area = Screen.FromControl(this).WorkingArea.Size; Size = new Size(Math.Min(Width, area.Width), Math.Min(Height, area.Height)); };
            healthTimer.Interval = 1000;
            healthTimer.Tick += delegate
            {
                if (recorder != null && !String.IsNullOrEmpty(recorder.LastError) && recorder.LastError != lastRecorderError)
                {
                    lastRecorderError = recorder.LastError;
                    collectionFault = true; UpdateGates();
                    Log("记录不完整：" + lastRecorderError);
                    status.Text = "记录出现错误；请停止实验并保留已取得数据：" + lastRecorderError;
                }
            };
            healthTimer.Start();
            UpdateGates();
        }

        private TableLayoutPanel Page(string title)
        {
            var tab = new TabPage(title) { Padding = new Padding(8), AutoScroll = true };
            var table = new TableLayoutPanel { Dock = DockStyle.Top, AutoSize = true, ColumnCount = 2, Padding = new Padding(2) };
            table.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize)); table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            tab.Controls.Add(table); tabs.TabPages.Add(tab); return table;
        }
        private void Row(TableLayoutPanel p, string label, Control control)
        {
            int i = p.RowCount++; p.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            var l = new Label { Text = label, AutoSize = true, MaximumSize = new Size(270, 0), Margin = new Padding(3, 8, 3, 7) };
            p.Controls.Add(l, 0, i); control.Dock = DockStyle.Fill; control.Margin = new Padding(3, 5, 3, 5); p.Controls.Add(control, 1, i);
        }
        private FlowLayoutPanel Flow(params Control[] controls)
        {
            var f = new FlowLayoutPanel { AutoSize = true, WrapContents = true, Margin = Padding.Empty };
            foreach (var c in controls) { c.Margin = new Padding(2, 3, 8, 3); f.Controls.Add(c); } return f;
        }
        private Button Button(string text, Action action)
        {
            var b = new Button { Text = text, AutoSize = true, MinimumSize = new Size(100, 30), Padding = new Padding(5, 1, 5, 1) };
            b.Click += delegate { try { action(); } catch (Exception ex) { ShowError(ex); } }; return b;
        }
        private Control Folder(TextBox box)
        {
            var t = new TableLayoutPanel { AutoSize = true, ColumnCount = 2, Dock = DockStyle.Fill };
            t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100)); t.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize)); box.Dock = DockStyle.Fill;
            t.Controls.Add(box, 0, 0); var browse = Button("选择", delegate { using (var d = new FolderBrowserDialog()) { d.SelectedPath = box.Text; if (d.ShowDialog(this) == DialogResult.OK) box.Text = d.SelectedPath; } });
            browse.MinimumSize = new Size(60, 26); t.Controls.Add(browse, 1, 0); return t;
        }
        private Label Note(string s) { return new Label { Text = s, AutoSize = true, MaximumSize = new Size(880, 0), ForeColor = Color.FromArgb(65, 65, 65) }; }
        private TextBox Small(TextBox box, int width) { box.Width = width; return box; }

        private void BuildObservePage()
        {
            var p = Page("1 观察与收集");
            Row(p, "会话保存位置", Folder(sessionRoot));
            createSession = Button("新建会话", NewSession); inventoryButton = Button("核对环境与文件", delegate { RunWork(CaptureInventory); });
            exportButton = Button("导出证据 ZIP", delegate { RunWork(Export); });
            Row(p, "会话", Flow(createSession, inventoryButton, exportButton, Button("检查已有会话", InspectPartial)));
            Row(p, "原厂 DSX-BSW 目录", Folder(installRoot));
            Row(p, "原厂图像输出目录", Folder(imageRoot)); Row(p, "原厂日志目录", Folder(logRoot));
            template.DropDownStyle = ComboBoxStyle.DropDownList; template.Width = 260;
            template.Items.AddRange(new object[] { "原厂二维拍照", "原厂多焦层：精细", "原厂多焦层：快速", "RDK 单步验证", "五层记录（人工操作）", "3×3×5 记录（人工操作）" }); template.SelectedIndex = 0;
            Row(p, "实验模板", template);
            var meta = new FlowLayoutPanel { AutoSize = true, WrapContents = true };
            foreach (string name in new[] { "倍率", "分辨率", "曝光", "Z起点µm", "Z终点µm", "Z间距µm", "层数", "备注" })
            {
                var f = new FlowLayoutPanel { AutoSize = true, WrapContents = false }; f.Controls.Add(new Label { Text = name, AutoSize = true, Margin = new Padding(1, 7, 4, 0) });
                var b = new TextBox { Width = name == "备注" ? 230 : 98 }; fields[name] = b; f.Controls.Add(b); meta.Controls.Add(f);
            }
            Row(p, "原厂实际参数", meta);
            startObserve = Button("开始观察", delegate { RunWork(StartObservation); }); stopObserve = Button("停止记录", delegate { RunWork(StopObservation); });
            Row(p, "记录控制", Flow(startObserve, stopObserve));
            var marks = new List<Control>(); foreach (string m in new[] { "开始操作", "原厂显示完成", "人工取消", "异常" }) { string mark = m; marks.Add(Button(m, delegate { Mark(mark); })); }
            Row(p, "人工时间标记", Flow(marks.ToArray()));
            Row(p, "记录含义", Note("只记录已有日志、文件变化和人工标记；文件稳定不代表曝光完成。停止记录不会停止相机或电机。先停止观察，再选择样本复制，减少采集时读盘影响。"));
            samples.Height = 112; samples.CheckOnClick = true; samples.HorizontalScrollbar = true;
            copyButton = Button("复制勾选样本并校验", delegate { RunWork(CopySelected); });
            Row(p, "本次观察到的文件", samples); Row(p, "样本", copyButton);
        }

        private void BuildControlPage()
        {
            var p = Page("2 受控验证（默认关闭）");
            Row(p, "用途", Note("这里会真实连接原厂接口。首次连接可能触发原厂初始化；连接、命令回复和位置缓存均不等于设备已停止或已到位。每次只执行一个人工选择的动作。"));
            optIn.Text = "启用本次受控验证"; normalStart.Text = "已核对正常原厂 RDK 启动方式"; stopReady.Text = "现场有人值守，已确认原厂停止手段";
            foreach (var c in new[] { optIn, normalStart, stopReady }) c.AutoSize = true;
            Row(p, "前置条件", Flow(optIn, normalStart, stopReady));
            host.Width = 190; host.PlaceholderTextCompat("填写环境报告中的本机地址");
            connectButton = Button("连接指定本机端点", delegate { RunWork(Connect); }); disconnectButton = Button("断开工具连接", Disconnect);
            Row(p, "本机地址 / 端口", Flow(host, port, connectButton, disconnectButton));
            initReady.Text = "原厂已完成初始化，已人工确认当前空闲"; initReady.AutoSize = true;
            Row(p, "初始化", initReady);
            Row(p, "命令期限（秒）", Flow(deadline, Note("到期只报告状态未知，不重发，也不代表已停止。")));
            takeControl = RdkButton("接管远程控制", async delegate { await Command("CHMODE=1"); remote = true; });
            releaseControl = RdkButton("释放回原厂", async delegate { await Command("CHMODE=0"); remote = false; });
            positionButton = RdkButton("读取 XY 与 Z 缓存", QueryPositions);
            errorButton = RdkButton("查询历史/共享错误", async delegate { await Command("GETERRCODE?"); });
            Row(p, "控制权 / 查询", Flow(takeControl, releaseControl, positionButton, errorButton));
            Row(p, "错误查询说明", Note("错误可能属于历史请求；原厂回复过程可能消耗共享错误状态。此按钮仅手动调用，不以‘无错’判本次成功。"));
            reported.ReadOnly = true; reported.Text = "尚未取得缓存位置"; Row(p, "报告位置（非到位判定）", reported);
            Row(p, "本次允许 X 范围 µm", Flow(Small(xmin, 110), Note("至"), Small(xmax, 110)));
            Row(p, "本次允许 Y 范围 µm", Flow(Small(ymin, 110), Note("至"), Small(ymax, 110)));
            Row(p, "本次允许 Z 范围 µm", Flow(Small(zmin, 110), Note("至"), Small(zmax, 110)));
            speed.DropDownStyle = ComboBoxStyle.DropDownList; speed.Width = 165; speed.Items.AddRange(new object[] { "速度枚举 0", "速度枚举 1", "速度枚举 2" });
            moveXY = RdkButton("单次 XY 绝对移动", MoveXY); moveZ = RdkButton("单次 Z 相对移动", MoveZ);
            Row(p, "XY目标 µm / 速度", Flow(Small(targetX, 100), Small(targetY, 100), speed, moveXY));
            Row(p, "Z相对量 µm", Flow(Small(deltaZ, 120), moveZ));
            Row(p, "位移要求", Note("必须先读取位置并填写经原厂确认的允许范围；仅接受整数µm目标。每次移动前显示起点缓存、目标及范围供人工核对。缓存未证明新鲜，本工具不输出‘到位成功’。"));
            Row(p, "快照独立保存目录", Folder(shotRoot));
            ordinarySnap.Text = "已在原厂确认普通2D、非时间序列、格式/曝光和标定"; ordinarySnap.AutoSize = true;
            Row(p, "成像条件", ordinarySnap);
            takeShot = RdkButton("单次普通二维拍照", Shot);
            shotStatus = RdkButton("读取拍照状态", async delegate { lastShotState = null; var reply = await Command("SHOTSTS?", currentCapture); lastShotState = reply.Value.Trim(); });
            stopShot = RdkButton("请求停止拍照", async delegate { lastShotState = null; await Command("ENDSHOT=0", currentCapture); Log("已发送停止拍照请求；未确认实际停止。请在原厂界面核对。"); });
            finishCapture = Button("记录本次拍照观察结束（人工）", FinishCaptureObservation);
            Row(p, "拍照验证", Flow(takeShot, shotStatus, stopShot, finishCapture));
            Row(p, "拍照完成", Note("SHOT最终回复只代表派发。拍照后锁住移动/下一次拍照；读取状态并在原厂核对后，人工记录观察结束。文件名可能带原厂后缀。此步骤不是自动采集验收。"));
        }

        private void BuildHelpPage()
        {
            var p = Page("3 明日操作卡");
            Row(p, "① 环境核对", Note("先解压整个ZIP，再双击程序。新建会话，选择DSX-BSW根目录，核对环境与34份关键文件。版本不同仍可观察，受控验证保持禁用。"));
            Row(p, "② 原厂观察", Note("选择原厂输出/日志目录，填写实际参数，开始观察。在原厂完成一次二维拍照及一次多焦层采集，使用人工时间标记。每个实验单独开始/停止记录。"));
            Row(p, "③ 受控验证", Note("原厂启动方式、现场停止手段确认后再启用第二页。监听表只表明端口存在，不证明可连接；没有正常RDK入口就保留报告，不尝试重启CTRL或绕过登录。"));
            Row(p, "④ 导出", Note("停止观察，勾选所需图像/日志样本并复制校验。确认原厂状态并释放控制后断开工具连接。导出ZIP；会话目录仍保留，可从日志中区分缺失、失败和人工判断。"));
            Row(p, "原厂日志", Note(@"默认候选：%PUBLIC%\OLYMPUS\DSX-BSW\Log\CurrentLog。只读取现有日志，不改变日志级别，不执行DSXLogBackup.vbs（该脚本会删除归档日志）。日志缓冲延迟不等于动作耗时。"));
            Row(p, "范围", Note("不修改FDM、驱动或标定；不自动对焦/回零/批量扫描。不使用相机驱动直接抢占设备。五层与45张模板只用于记录人工实验。"));
            Row(p, "验证边界", Note("已在Windows虚拟机用模拟服务/文件进行工具验证；真实Win10 1909、DSX运动反馈、曝光对应和异常恢复仍需现场确认。"));
        }

        private Button RdkButton(string label, Func<Task> action)
        { var b = Button(label, delegate { RunWork(action); }); rdkActions.Add(b); return b; }
        private void NewSession()
        {
            if (observing || capturePending || (rdk != null && rdk.Connected)) throw new InvalidOperationException("先记录未结束拍照的正常或异常结案，再停止观察并核对原厂状态后断开连接。");
            if (recorder != null) { recorder.Dispose(); recorder = null; }
            if (rdk != null) { rdk.Dispose(); rdk = null; }
            if (session != null) session.Dispose();
            session = new EvidenceSession(sessionRoot.Text); sealedSession = false; identityMatched = false; samples.Items.Clear(); collectionFault = false; lastRecorderError = null; capturePending = false; remote = false;
            session.Add("tool", "tool_version", new { version = Program.Version, mode = "passive_default" });
            Log("新会话：" + session.DirectoryPath); UpdateGates();
        }
        private async Task CaptureInventory()
        {
            EnsureSession(); string root = installRoot.Text;
            var result = await Task.Run(delegate { return Inventory.Capture(root, Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "baseline.json")); });
            inventoryRoot = result.Root; identityMatched = result.Matched;
            Util.WriteJson(Path.Combine(session.DirectoryPath, "environment.json"), result.Data);
            session.Add("inventory", "snapshot", result.Data); session.SaveSummary("recording");
            Log(identityMatched ? "关键文件身份匹配；受控验证还需现场前置条件。" : "身份未匹配或有缺失；继续被动观察，受控验证禁用。");
        }
        private async Task StartObservation()
        {
            EnsureSession(); if (observing) return;
            if (capturePending) throw new InvalidOperationException("先记录本次拍照的正常或异常观察结束，不能切换实验身份。");
            var parameters = new Dictionary<string, object>(); foreach (var item in fields) parameters[item.Key] = item.Value.Text;
            parameters["source"] = "operator"; parameters["template"] = template.Text;
            session.NewRun(template.Text, parameters);
            string images = imageRoot.Text, logs = logRoot.Text;
            await Task.Run(delegate
            {
                if (recorder != null) recorder.Dispose();
                recorder = new ObservationRecorder(session, images, logs); recorder.Start();
            });
            observing = true; collectionFault = false; lastRecorderError = null;
            samples.Items.Clear(); Log("观察已开始；原厂操作由现场人员完成。"); UpdateGates();
        }
        private async Task StopObservation()
        {
            await Task.Run(delegate { if (recorder != null) recorder.Stop(); }); observing = false;
            if (recorder != null) { samples.Items.Clear(); foreach (string f in recorder.ObservedFiles) samples.Items.Add(f, false); }
            if (session != null) { session.Add("operator", "observation_stopped", new { device_stopped = "not_asserted" }); session.SaveSummary("recording"); }
            Log("记录已停止。未向设备发送停止命令。"); UpdateGates();
        }
        private void Mark(string text) { EnsureSession(); session.Add("operator", "marker", new { label = text, hardware_confirmation = false }); Log("人工标记：" + text); }
        private async Task CopySelected()
        {
            EnsureSession(); if (recorder == null || observing) throw new InvalidOperationException("先停止观察。");
            var list = samples.CheckedItems.Cast<string>().ToArray(); if (list.Length == 0) throw new InvalidOperationException("请勾选需要带回的样本。");
            await Task.Run(delegate { recorder.CopySamples(list); }); Log("样本复制处理完成；逐项成功/失败见事件记录。");
        }
        private async Task Export()
        {
            EnsureSession(); if (capturePending || observing || (rdk != null && rdk.Connected)) throw new InvalidOperationException("先记录拍照观察结案、停止观察、核对原厂状态并断开连接。");
            if (rdk != null) { rdk.Dispose(); rdk = null; }
            if (recorder != null) { recorder.Dispose(); recorder = null; }
            string path = await Task.Run(delegate { return session.ExportZip(); }); sealedSession = true;
            Log("已导出：" + path); MessageBox.Show(this, path + "\r\n会话目录仍保留。下一次实验请新建会话。", "证据包已导出");
        }
        private void InspectPartial()
        {
            if (working || observing || (rdk != null && rdk.Connected)) throw new InvalidOperationException("请先结束当前操作并断开工具连接，再检查已有会话。");
            using (var dialog = new FolderBrowserDialog())
            {
                dialog.SelectedPath = sessionRoot.Text;
                if (dialog.ShowDialog(this) != DialogResult.OK) return;
                var result = EvidenceSession.InspectRecovery(dialog.SelectedPath);
                MessageBox.Show(this, Util.Json(result) + "\r\n这是只读检查；不会修复或续写历史记录。请将整个原会话文件夹一并带回。", "已有会话检查");
            }
        }
        private void EnsureSession() { if (session == null || sealedSession) throw new InvalidOperationException("请先新建未封存的会话。"); }
        private async Task Connect()
        {
            EnsureSession(); if (capturePending || !identityMatched || !optIn.Checked || !normalStart.Checked || !stopReady.Checked || !observing || collectionFault) throw new InvalidOperationException("请先结案上次拍照观察，完成文件身份和现场前置条件核对，并开始正常观察记录。");
            string address = host.Text.Trim(); System.Net.IPAddress ip;
            if (!System.Net.IPAddress.TryParse(address, out ip)) throw new InvalidOperationException("请填写监听报告中的本机IP地址，不使用主机名猜测。");
            bool local = System.Net.IPAddress.IsLoopback(ip) || System.Net.NetworkInformation.NetworkInterface.GetAllNetworkInterfaces().SelectMany(x => x.GetIPProperties().UnicastAddresses).Any(x => x.Address.Equals(ip));
            if (!local) throw new InvalidOperationException("本工具只允许同机原厂端点；填写的IP不属于本机。");
            if (!Confirm("连接可能触发原厂初始化。确认原厂处于适合连接的状态，且当前没有其他RDK客户端？")) return;
            if (rdk != null) rdk.Dispose();
            EvidenceSession boundSession = session;
            rdk = new RdkClient(delegate(string source, string kind, object data, string request, string capture)
            { RecordRdk(boundSession, source, kind, data, request, capture); });
            rdk.StateChanged += delegate { if (IsHandleCreated && !IsDisposed) BeginInvoke((Action)UpdateGates); };
            remote = false; capturePending = false; initReady.Checked = false; reportedX = reportedY = reportedZ = null;
            await rdk.ConnectAsync(address, (int)port.Value, 5000); Log("已建立连接；等待/核对原厂初始化完成，不自动发送INIT。");
        }
        private void Disconnect()
        {
            if (rdk == null) return;
            if (rdk.Connected && !Confirm("断开工具连接不会停止电机或保证释放原厂控制。已核对原厂状态，并完成可执行的释放流程？")) return;
            rdk.Disconnect(); remote = false; initReady.Checked = false;
            if (session != null) session.Add("operator", "disconnect_acknowledged", new { device_stopped = "not_asserted" });
            Log("工具连接已断开；设备状态须在原厂核对。"); UpdateGates();
        }
        private void RecordRdk(EvidenceSession targetSession, string source, string kind, object data, string request, string capture)
        {
            targetSession.Add(source, kind, data, request, capture);
            if (IsHandleCreated && !IsDisposed) BeginInvoke((Action)delegate { Log(kind + " " + Util.Json(data)); });
        }
        private async Task<RdkReply> Command(string command, string captureId = null, string requestId = null)
        {
            EnsureSession(); if (rdk == null || !rdk.Connected || rdk.Unknown || !initReady.Checked) throw new InvalidOperationException("原厂连接或初始化尚未确认。");
            if (!optIn.Checked || !normalStart.Checked || !stopReady.Checked || !identityMatched || collectionFault) throw new InvalidOperationException("受控验证前置条件已失效，未派发命令。");
            var reply = await rdk.SendAsync(command, (int)deadline.Value * 1000, requestId ?? Guid.NewGuid().ToString("N"), captureId);
            if (reply.IsError) throw new InvalidOperationException("原厂错误回复：" + reply.Raw);
            Log("协议回复：" + reply.Raw + "（不代表物理到位或图像完成）"); return reply;
        }
        private async Task QueryPositions()
        {
            reportedX = reportedY = reportedZ = null; reported.Text = "正在读取；旧缓存已作废。";
            var xy = await Command("STGPOS?"); string[] parts = xy.Value.Trim().Split(',');
            var z = await Command("ZPOS?"); decimal xVal, yVal, zVal;
            reportedX = reportedY = reportedZ = null;
            if (parts.Length == 2 && decimal.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out xVal) && decimal.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out yVal)) { reportedX = xVal; reportedY = yVal; }
            if (decimal.TryParse(z.Value.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out zVal)) reportedZ = zVal;
            reported.Text = "UI缓存：XY=" + xy.Value + " ; Z=" + z.Value + " µm；鲜度未证";
            session.Add("rdk", "reported_position", new { x_um = reportedX, y_um = reportedY, z_um = reportedZ, evidence_kind = "ui_cache", freshness = "unproven" });
        }
        private decimal Whole(TextBox box, string label)
        {
            decimal v; if (!decimal.TryParse(box.Text.Trim(), NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out v) || v != decimal.Truncate(v) || v < int.MinValue || v > int.MaxValue) throw new InvalidOperationException(label + "必须填写整数µm；不会静默取整。"); return decimal.Truncate(v);
        }
        private void Range(decimal current, decimal target, TextBox min, TextBox max, string axis)
        {
            decimal lo = Whole(min, axis + "下限"), hi = Whole(max, axis + "上限");
            if (lo > hi || current < lo || current > hi || target < lo || target > hi) throw new InvalidOperationException(axis + "起点缓存或目标不在本次允许范围内。");
        }
        private async Task MoveXY()
        {
            if (capturePending || !observing || collectionFault) throw new InvalidOperationException("需要正常观察记录，且上次拍照观察已结束。");
            if (!reportedX.HasValue || !reportedY.HasValue) throw new InvalidOperationException("先读取并在原厂核对位置缓存。");
            decimal x = Whole(targetX, "X目标"), y = Whole(targetY, "Y目标");
            Range(reportedX.Value, x, xmin, xmax, "X"); Range(reportedY.Value, y, ymin, ymax, "Y");
            if (speed.SelectedIndex < 0) throw new InvalidOperationException("请选择经现场核对的原厂速度枚举。");
            if (!Confirm("单次XY移动：缓存(" + reportedX + "," + reportedY + ") → 目标(" + x + "," + y + ") µm。缓存鲜度未证；已在原厂核对目标/范围与可用停止手段？")) return;
            string requestId = Guid.NewGuid().ToString("N");
            session.Add("operator", "move_target", new { x_um = x, y_um = y, source = "operator", physical_completion = "not_asserted" }, requestId);
            try { await Command("MVSTG=" + x.ToString(CultureInfo.InvariantCulture) + "," + y.ToString(CultureInfo.InvariantCulture) + "," + speed.SelectedIndex, null, requestId); }
            finally { reportedX = reportedY = null; reported.Text = "移动后请重新读取并核对缓存位置；未判定到位。"; }
        }
        private async Task MoveZ()
        {
            if (capturePending || !observing || collectionFault) throw new InvalidOperationException("需要正常观察记录，且上次拍照观察已结束。");
            if (!reportedZ.HasValue || reportedZ.Value != decimal.Truncate(reportedZ.Value)) throw new InvalidOperationException("先读取整数µm的Z缓存并在原厂核对；小数起点不自动取整。");
            decimal delta = Whole(deltaZ, "Z相对量"), target = reportedZ.Value + delta; if (delta == 0) throw new InvalidOperationException("Z相对量不能为0。");
            Range(reportedZ.Value, target, zmin, zmax, "Z");
            if (!Confirm("单次Z相对移动：缓存" + reportedZ + " + " + delta + " → 目标" + target + " µm。已在原厂核对方向、范围和停止手段？")) return;
            string requestId = Guid.NewGuid().ToString("N");
            session.Add("operator", "move_target", new { z_um = target, delta_um = delta, source = "operator", physical_completion = "not_asserted" }, requestId);
            try { await Command("MVZR=" + delta.ToString(CultureInfo.InvariantCulture), null, requestId); }
            finally { reportedZ = null; reported.Text = "移动后请重新读取并核对缓存位置；未判定到位。"; }
        }
        private async Task Shot()
        {
            if (!observing) throw new InvalidOperationException("请先在观察页开始记录，以保留图像和日志变化。");
            if (capturePending || !ordinarySnap.Checked) throw new InvalidOperationException("请完成上次观察，并核对普通二维成像设置。");
            string path = Path.GetFullPath(shotRoot.Text.Trim());
            if (path.Any(c => c > 127 || char.IsWhiteSpace(c) || c == ',' || c == '"') || path.StartsWith(@"\\", StringComparison.Ordinal)) throw new InvalidOperationException("RDK保存目录须为本地ASCII路径，不能有空格/逗号/引号；请另选简单目录。");
            string observedPath = Path.GetFullPath(imageRoot.Text.Trim()).TrimEnd(Path.DirectorySeparatorChar);
            if (!path.TrimEnd(Path.DirectorySeparatorChar).Equals(observedPath, StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("快照保存根目录必须与当前观察的图像目录相同。");
            if (!Confirm("将选择普通Snap模式并请求一张二维图；不改变曝光/分辨率/格式。确认原厂设置、当前位置和保存目录？")) return;
            currentCapture = Guid.NewGuid().ToString("N"); string dir = Path.Combine(path, "capture_" + currentCapture); Directory.CreateDirectory(dir);
            await Command("ACQMODE=1", currentCapture);
            if (!observing || collectionFault || !ordinarySnap.Checked) throw new InvalidOperationException("记录或成像前置条件已失效，未派发SHOT。");
            capturePending = true; lastShotState = null;
            session.Add("operator", "capture_requested", new { capture_id = currentCapture, directory = dir, expected_name_family = "frame", image_complete = "not_asserted" }, null, currentCapture);
            await Command("SHOT=1," + Path.Combine(dir, "frame"), currentCapture);
        }
        private void FinishCaptureObservation()
        {
            bool abnormal = collectionFault || rdk == null || !rdk.Connected || rdk.Unknown;
            if (!capturePending || (!abnormal && lastShotState != "0")) throw new InvalidOperationException("先读取拍照状态并在原厂核对；异常状态可人工结束为未确认结果。");
            if (!Confirm((abnormal ? "这是异常结案，工具未确认设备停止。" : "") + "已在原厂核对本次拍照结束/停止及当前位置？这仅记录人工观察，不认定图像成功或电机到位。")) return;
            session.Add("operator", "capture_observation_ended", new { reported_state = lastShotState, abnormal = abnormal, image_success = "not_asserted", device_stopped = "not_asserted" }, null, currentCapture);
            capturePending = false; currentCapture = null; UpdateGates();
        }
        private bool Confirm(string message) { return MessageBox.Show(this, message, "核对本次动作", MessageBoxButtons.OKCancel, MessageBoxIcon.Information, MessageBoxDefaultButton.Button2) == DialogResult.OK; }
        private async void RunWork(Func<Task> action)
        {
            if (working) return; working = true; UpdateGates();
            try { await action(); } catch (Exception ex) { ShowError(ex); }
            finally { working = false; UpdateGates(); }
        }
        private void ShowError(Exception ex)
        {
            try { if (session != null && !sealedSession) session.AddError("ui_operation", ex); } catch { }
            Log("未完成：" + ex.Message); MessageBox.Show(this, Util.Redact(ex.Message), "操作未完成", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
        private void Log(string message)
        { activity.AppendText(DateTime.Now.ToString("HH:mm:ss") + "  " + Util.Redact(message) + Environment.NewLine); if (activity.TextLength > 100000) activity.Text = activity.Text.Substring(activity.TextLength - 70000); }
        private void UpdateGates()
        {
            bool has = session != null && !sealedSession && !session.IsClosed;
            bool connected = rdk != null && rdk.Connected;
            bool ready = has && connected && !rdk.Unknown && !rdk.Busy && !working && !collectionFault && initReady.Checked && optIn.Checked && normalStart.Checked && stopReady.Checked && identityMatched;
            createSession.Enabled = !working && !observing && !connected && !capturePending;
            inventoryButton.Enabled = has && !working && !observing && !connected;
            startObserve.Enabled = has && !working && !observing && !capturePending; stopObserve.Enabled = observing && !working;
            exportButton.Enabled = session != null && !sealedSession && !working && !observing && !connected && !capturePending; copyButton.Enabled = has && !working && !observing && recorder != null;
            connectButton.Enabled = has && !working && !connected && !capturePending && identityMatched && optIn.Checked && normalStart.Checked && stopReady.Checked && observing && !collectionFault;
            disconnectButton.Enabled = connected && !working;
            foreach (var b in rdkActions) b.Enabled = ready;
            takeControl.Enabled = ready && !remote; releaseControl.Enabled = ready && remote && !capturePending;
            moveXY.Enabled = moveZ.Enabled = ready && remote && !capturePending && observing;
            positionButton.Enabled = ready && remote && !capturePending;
            errorButton.Enabled = ready && !capturePending;
            takeShot.Enabled = ready && remote && !capturePending && ordinarySnap.Checked && observing;
            shotStatus.Enabled = ready && remote; stopShot.Enabled = ready && remote && capturePending;
            finishCapture.Enabled = has && !working && capturePending && (lastShotState == "0" || collectionFault || rdk == null || !rdk.Connected || rdk.Unknown);
            optIn.Enabled = normalStart.Enabled = stopReady.Enabled = !working && !connected;
            initReady.Enabled = connected && !working; ordinarySnap.Enabled = !working;
            installRoot.Parent.Enabled = !working && !connected && !observing; imageRoot.Parent.Enabled = !observing && !working; logRoot.Parent.Enabled = !observing && !working;
            status.Text = (working ? "正在处理… " : "") + (rdk != null && rdk.Unknown ? "RDK状态未知：不再派发命令，请核对原厂。" : connected ? "受控连接已建立；协议回复不等于物理到位。" : "观察模式：未建立RDK连接。") + (observing ? " 正在记录。" : "");
        }
        private void OnClosing(object sender, FormClosingEventArgs e)
        {
            if (working) { e.Cancel = true; MessageBox.Show(this, "当前工作尚未结束。命令超时后可核对原厂并关闭；关闭程序不等于停机。", "正在处理"); return; }
            if (rdk != null && rdk.Connected && !Confirm("退出只关闭工具连接，不会停止设备。已核对原厂状态并完成必要释放？")) { e.Cancel = true; return; }
            healthTimer.Stop(); healthTimer.Dispose();
            try { if (recorder != null) recorder.Dispose(); if (rdk != null) rdk.Dispose(); if (session != null) { if (!sealedSession) session.SaveSummary("closed_partial"); session.Dispose(); } } catch { }
        }
        public void SaveUiSmoke(string directory)
        {
            var rows = new List<object>();
            foreach (Size size in new[] { new Size(1280, 720), new Size(1366, 768) })
            {
                Size = size;
                for (int i = 0; i < tabs.TabPages.Count; i++)
                {
                    tabs.SelectedIndex = i; Application.DoEvents();
                    using (var bitmap = new Bitmap(Width, Height)) { DrawToBitmap(bitmap, new Rectangle(0, 0, Width, Height)); bitmap.Save(Path.Combine(directory, "ui-" + size.Width + "-tab" + i + ".png")); }
                    float dpi; using (Graphics graphics = CreateGraphics()) dpi = graphics.DpiX;
                    rows.Add(new { width = Width, height = Height, tab = i, dpi_x = dpi, screenshot = true, default_connect_enabled = connectButton.Enabled, default_move_enabled = moveXY.Enabled });
                    tabs.TabPages[i].AutoScrollPosition = new Point(0, 10000); Application.DoEvents();
                    using (var bitmap = new Bitmap(Width, Height)) { DrawToBitmap(bitmap, new Rectangle(0, 0, Width, Height)); bitmap.Save(Path.Combine(directory, "ui-" + size.Width + "-tab" + i + "-bottom.png")); }
                    tabs.TabPages[i].AutoScrollPosition = Point.Empty;
                }
            }
            Util.WriteJson(Path.Combine(directory, "ui-smoke.json"), rows);
            if (connectButton.Enabled || moveXY.Enabled || takeShot.Enabled) throw new InvalidOperationException("Default control gates unexpectedly enabled.");
        }
    }
    internal static class LegacyTextBoxExtensions
    { public static void PlaceholderTextCompat(this TextBox box, string text) { new ToolTip().SetToolTip(box, text); } }
}
