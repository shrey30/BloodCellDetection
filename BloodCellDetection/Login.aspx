<%@ Page Title="" Language="C#" MasterPageFile="~/LoginMaster.Master" AutoEventWireup="true" CodeBehind="Login.aspx.cs" Inherits="BloodCellDetection.Login" %>


<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #74ebd5, #9face6);
            margin: 0;
            padding: 0;
        }

        .login-box {
            width: 350px;
            margin: 100px auto;
            padding: 30px;
            background: #fff;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            text-align: center;
            animation: fadeIn 1s ease-in-out;
        }

        .login-box h2 {
            margin-bottom: 20px;
            font-size: 24px;
            color: #333;
        }

        .form-control {
            width: 100%;
            padding: 10px;
            margin: 8px 0 15px 0;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 14px;
            transition: border 0.3s;
        }

        .form-control:focus {
            border: 1px solid #6c63ff;
            outline: none;
        }

        .btn {
            width: 100%;
            padding: 10px;
            background: #6c63ff;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            color: white;
            cursor: pointer;
            transition: background 0.3s;
        }

        .btn:hover {
            background: #574b90;
        }

        .error-label {
            display: block;
            margin-top: 10px;
            font-size: 13px;
            color: red;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</asp:Content>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <div class="login-box">
        <h2>User Login</h2>

        <asp:TextBox ID="txtUserId" runat="server" CssClass="form-control" placeholder="Enter User ID"></asp:TextBox>

        <asp:TextBox ID="txtPassword" runat="server" TextMode="Password" CssClass="form-control" placeholder="Enter Password"></asp:TextBox>

        <asp:Button ID="btnLogin" runat="server" Text="Login" OnClick="btnLogin_Click" CssClass="btn" />

        <asp:Label ID="lblMessage" runat="server" CssClass="error-label"></asp:Label>
    </div>
</asp:Content>

