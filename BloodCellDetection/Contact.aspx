<%@ Page Title="Contact Us" Language="C#" MasterPageFile="~/HomeMaster.Master" AutoEventWireup="true" CodeBehind="Contact.aspx.cs" Inherits="BloodCellDetection.Contact" %>

<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
        }

        .contact-container {
            width: 80%;
            margin: 50px auto;
            text-align: center;
        }

        .contact-title {
            font-size: 28px;
            font-weight: bold;
            color: #0d47a1;
            margin-bottom: 30px;
        }

        .contact-grid {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
        }

        .contact-card {
            background: #ffffff;
            width: 300px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
            text-align: center;
            transition: transform 0.3s;
        }

        .contact-card:hover {
            transform: translateY(-5px);
        }

        .contact-card h3 {
            color: #0d47a1;
            font-size: 20px;
            margin-bottom: 5px;
        }

        .contact-card p {
            color: #444;
            margin: 5px 0;
        }

        .contact-card .role {
            font-weight: bold;
            color: #1976d2;
        }

        .contact-card .icon {
            font-size: 40px;
            color: #1976d2;
            margin-bottom: 10px;
        }

        .contact-card a {
            color: #0d47a1;
            text-decoration: none;
        }

        .contact-card a:hover {
            text-decoration: underline;
        }
    </style>
</asp:Content>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <div class="contact-container">
        <div class="contact-title">Contact Our Team</div>

        <div class="contact-grid">
            <!-- Contact 1 -->
            <div class="contact-card">
                <div class="icon">👨‍⚕️</div>
                <h3>Dr. Arjun Patel</h3>
                <p class="role">Chief Pathologist</p>
                <p>Email: <a href="mailto:arjun.patel@bloodlab.com">arjun.patel@bloodlab.com</a></p>
                <p>Phone: <a href="tel:+11234567890">+1 123-456-7890</a></p>
            </div>

            <!-- Contact 2 -->
            <div class="contact-card">
                <div class="icon">🧪</div>
                <h3>Neha Sharma</h3>
                <p class="role">Lab Technician</p>
                <p>Email: <a href="mailto:neha.sharma@bloodlab.com">neha.sharma@bloodlab.com</a></p>
                <p>Phone: <a href="tel:+19876543210">+1 987-654-3210</a></p>
            </div>

            <!-- Contact 3 -->
            <div class="contact-card">
                <div class="icon">💬</div>
                <h3>Ravi Kumar</h3>
                <p class="role">Support & Appointments</p>
                <p>Email: <a href="mailto:support@bloodlab.com">support@bloodlab.com</a></p>
                <p>Phone: <a href="tel:+10123456789">+1 012-345-6789</a></p>
            </div>
        </div>
    </div>
</asp:Content>
