<%@ Page Title="" Language="C#" MasterPageFile="~/HomeMaster.Master" AutoEventWireup="true" CodeBehind="History.aspx.cs" Inherits="BloodCellDetection.History" %>
<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
        }

        .history-container {
            width: 75%;
            margin: 40px auto;
            background: #fff;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 10px;
            padding: 25px;
        }

        .history-title {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #0d47a1;
            margin-bottom: 20px;
        }

        table.history-table {
            width: 100%;
            border-collapse: collapse;
            text-align: center;
        }

        table.history-table th, table.history-table td {
            border: 1px solid #ddd;
            padding: 10px;
        }

        table.history-table th {
            background-color: #0d47a1;
            color: #fff;
            font-weight: bold;
        }

        table.history-table tr:nth-child(even) {
            background-color: #f2f6fc;
        }

        table.history-table tr:hover {
            background-color: #e3f2fd;
        }

        .btn-view {
            background-color: #0d47a1;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            cursor: pointer;
            font-size: 13px;
            transition: 0.3s;
        }

        .btn-view:hover {
            background-color: #1976d2;
        }

        .info {
            text-align: center;
            color: #666;
            margin-top: 15px;
        }
    </style>
</asp:Content>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <div class="history-container">
        <div class="history-title">Report History</div>
        <asp:Literal ID="litHistory" runat="server"></asp:Literal>
    </div>
</asp:Content>