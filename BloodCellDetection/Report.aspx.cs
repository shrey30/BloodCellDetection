using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Xml.Linq;

namespace BloodCellDetection
{
    public partial class Report : System.Web.UI.Page
    {
        private string latestFilePath = string.Empty;

        protected void Page_Load(object sender, EventArgs e)
        {
            if (!IsPostBack)
                ShowLatestReport();
        }

        private void ShowLatestReport()
        {
            if (Session["UserName"] == null)
                Session["UserName"] = "JohnDoe";

            string username = Session["UserName"].ToString();
            string folderPath = Server.MapPath("~/App_Data/Reports/");

            if (!Directory.Exists(folderPath))
            {
                litReport.Text = "<p class='info'>No reports found.</p>";
                return;
            }

            // Find the latest report for the user
            var files = new DirectoryInfo(folderPath)
                .GetFiles($"{username}_*.xml")
                .OrderByDescending(f => f.CreationTime)
                .ToList();

            if (files.Count == 0)
            {
                litReport.Text = "<p class='info'>No reports available for this user.</p>";
                return;
            }

            latestFilePath = files.First().FullName;

            // Load XML
            XElement reportXml = XElement.Load(latestFilePath);
            string uname = reportXml.Attribute("UserName")?.Value ?? "";
            string reportId = reportXml.Attribute("ReportID")?.Value ?? "";
            string date = reportXml.Attribute("DateTime")?.Value ?? "";
            var tests = reportXml.Element("Tests")?.Elements("Test");
            string diagnosis = reportXml.Element("Diagnosis")?.Value ?? "";

            // Build HTML table
            string html = $@"
            <div class='info'>
            <h1>Patient Name :{uname}</h1> <br />
                <strong>Report ID:</strong> {reportId}<br />
                <strong>Date:</strong> {date}
            </div>
            <table class='report-table'>
                <tr><th>Test Name</th><th>Result</th><th>Normal Range</th></tr>";

            foreach (var test in tests)
            {
                html += $"<tr>" +
                        $"<td>{test.Element("TestName")?.Value}</td>" +
                        $"<td>{test.Element("Result")?.Value}</td>" +
                        $"<td>{test.Element("NormalRange")?.Value}</td>" +
                        $"</tr>";
            }

            html += "</table>";
            html += $"<div class='diagnosis'><b>Dignosis:</b>{diagnosis}</div>";

            litReport.Text = html;
        }

        protected void btnDownload_Click(object sender, EventArgs e)
        {
            string username = Session["UserName"]?.ToString() ?? "Unknown";
            string folderPath = Server.MapPath("~/App_Data/Reports/");

            var files = new DirectoryInfo(folderPath)
                .GetFiles($"{username}_*.xml")
                .OrderByDescending(f => f.CreationTime)
                .ToList();

            if (files.Count == 0)
            {
                Response.Write("<script>alert('No report available to download.');</script>");
                return;
            }

            string latestFile = files.First().FullName;

            Response.ContentType = "application/xml";
            Response.AppendHeader("Content-Disposition", $"attachment; filename={Path.GetFileName(latestFile)}");
            Response.TransmitFile(latestFile);
            Response.End();
        }
    }
}