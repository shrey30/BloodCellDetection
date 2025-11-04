<%@ Page Title="" Language="C#" MasterPageFile="~/HomeMaster.Master" AutoEventWireup="true" CodeBehind="UHome.aspx.cs" Inherits="BloodCellDetection.UHome" %>
<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>
<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <div class="hero">
  <h1>Advanced Blood Cell Detection</h1>
  <p>Upload microscopic images for real-time blood cell analysis,<br> classification, and infection diagnosis using AI.</p>
  <asp:Button ID="btnUpload" runat="server" Text="Upload Sample" PostBackUrl="~/Dashboard.aspx" />
</div>

<div class="cards">
  <div class="card">
    <h3>Accurate Detection</h3>
    <p>Identifies and classifies blood cells with high precision.</p>
  </div>
  <div class="card">
    <h3>Graphical Analysis</h3>
    <p>Visual representation of blood cell distribution using charts.</p>
  </div>
  <div class="card">
    <h3>Instant Reports</h3>
    <p>Download comprehensive reports for diagnosis.</p>
  </div>
</div>
</asp:Content>
