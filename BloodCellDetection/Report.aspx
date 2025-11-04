<%@ Page Title="" Language="C#" MasterPageFile="~/HomeMaster.Master" AutoEventWireup="true" CodeBehind="Report.aspx.cs" Inherits="BloodCellDetection.Report" %>
<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
        }

        .report-container {
            width: 100%;
            margin: 40px auto;
            background: #fff;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 10px;
            padding: 20px 30px;
        }

        .report-title {
            text-align: center;
            font-size: 24px;
            color: #0d47a1;
            font-weight: bold;
            margin-bottom: 20px;
        }

        table.report-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }

        table.report-table th, table.report-table td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }

        table.report-table th {
            background-color: #0d47a1;
            color: #fff;
            font-weight: bold;
        }

        table.report-table tr:nth-child(even) {
            background-color: #f2f6fc;
        }

        .diagnosis {
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            margin-top: 20px;
            color: #006400;
        }

        .button-container {
            text-align: center;
            margin-top: 25px;
        }

        .btn {
            background-color: #0d47a1;
            color: white;
            padding: 10px 20px;
            margin: 0 8px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: 0.3s;
        }

        .btn:hover {
            background-color: #1976d2;
        }

        .info {
            text-align: center;
            color: #555;
            margin-bottom: 10px;
        }
    </style>
</asp:Content>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <div class="report-container">
        <div class="report-title">Latest Blood Test Report</div>
        <asp:Literal ID="litReport" runat="server"></asp:Literal>

        <div class="button-container">
            <asp:Button ID="btnDownload" runat="server" Text="⬇️ Download Report" CssClass="btn" OnClick="btnDownload_Click" />
            <asp:Button ID="btnShare" runat="server" Text="🔗 Share Report" CssClass="btn" OnClientClick="shareReport(); return false;" />
        </div>
    </div>

    <script>
        function shareReport() {
            const url = window.location.href;
            navigator.clipboard.writeText(url);
            alert("✅ Report link copied to clipboard:\n" + url);
        }
    </script>
</asp:Content>