<%@ Page Title="" Language="C#" MasterPageFile="~/HMaster.Master" AutoEventWireup="true" CodeBehind="Home.aspx.cs" Inherits="BloodCellDetection.Home" %>
<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>
<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <table style="width: 100%;">
        <tr>
            <td>Select Image to upload</td>
            <td>
                <asp:FileUpload ID="FileUpload1" runat="server" />
                <br />
                <asp:Label ID="lblMessage" runat="server" Text="Label"></asp:Label>
            </td>
            <td>
                <asp:Button ID="Button1" runat="server" Text="Upload" OnClick="Button1_Click" />
            </td>
        </tr>
        <tr>
            <td>&nbsp;</td>
            <td>
                <asp:Image ID="Image1" runat="server" Height="300px" Width="400px" />
            </td>
            <td>&nbsp;</td>
        </tr>
        <tr>
            <td>&nbsp;</td>
            <td>&nbsp;</td>
            <td>&nbsp;</td>
        </tr>
    </table>
</asp:Content>
