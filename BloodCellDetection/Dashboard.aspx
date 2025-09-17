<%@ Page Title="" Language="C#" MasterPageFile="~/HomeMaster.Master" AutoEventWireup="true" CodeBehind="Dashboard.aspx.cs" Inherits="BloodCellDetection.Dashboard" %>

<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">

    <style>
        .upload-box {
            width: 100%;
            margin: 20px auto;
            padding: 10px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            background: #fff;
            text-align: center;
        }

            .upload-box h2 {
                margin-bottom: 20px;
                font-size: 22px;
                font-weight: 600;
                color: #333;
            }

        .upload-controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }

        .upload-box .btn {
            padding: 8px 20px;
            font-size: 15px;
            border-radius: 6px;
            cursor: pointer;
        }

        .upload-box .message {
            margin-top: 10px;
            font-weight: bold;
            color: green;
            display: block;
        }

        .upload-box .filename {
            margin-top: 5px;
            color: #007bff;
            font-style: italic;
            word-break: break-all;
        }

        .upload-box img {
            margin-top: 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
            max-width: 100%;
            height: auto;
        }
    </style>
</asp:Content>
<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <div class="upload-box">
        <h2>Upload Your Image</h2>

        <div class="upload-controls">
            <asp:FileUpload ID="FileUpload1" runat="server" CssClass="form-control" />
            <asp:Button ID="Button1" runat="server" Text="Upload"
                OnClick="Button1_Click" CssClass="btn btn-primary" />

        </div>

        <asp:Label ID="lblMessage" runat="server" CssClass="message"></asp:Label>
        <asp:Label ID="lblFileName" runat="server" CssClass="filename"></asp:Label>
        <asp:Image ID="Image1" runat="server" Height="250px" Width="350px" />
        <br />
        <asp:Button ID="Button2" runat="server" Text="Detect"
            CssClass="btn btn-primary" OnClick="Button2_Click" />
    </div>

    
    <asp:Panel ID="Panel1" runat="server">


        <!-- Image 1 -->
        <div class="upload-box">
            <a href="#img1">
                <asp:Image ID="Image2" runat="server" ImageUrl="~/Python/step.jpg" CssClass="zoomable" Width="100%" />
            </a>
        </div>

        <!-- Image 2 -->
        <div class="upload-box">
            <a href="#img2">
                <asp:Image ID="Image3" runat="server" ImageUrl="~/Python/final.jpg" CssClass="zoomable" Width="100%" />
            </a>
        </div>

        <!-- Image 3 -->
        <div class="upload-box">
            <a href="#img3">
                <asp:Image ID="Image4" runat="server" ImageUrl="~/Python/out.png" CssClass="zoomable" Width="100%" />
            </a>
        </div>
    </asp:Panel>
</asp:Content>
