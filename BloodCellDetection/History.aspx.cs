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
    public partial class History : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            if (!IsPostBack)
                LoadReportHistory();
        }

        private void LoadReportHistory()
        {
            if (Session["UserName"] == null)
                Session["UserName"] = "JohnDoe"; // Example user

            string username = Session["UserName"].ToString();
            string folderPath = Server.MapPath("~/App_Data/Reports/");

            if (!Directory.Exists(folderPath))
            {
                litHistory.Text = "<p class='info'>No report history found.</p>";
                return;
            }

            // Get all XML files for this user
            var files = new DirectoryInfo(folderPath)
                .GetFiles($"{username}_*.xml")
                .OrderByDescending(f => f.CreationTime)
                .ToList();

            if (files.Count == 0)
            {
                litHistory.Text = "<p class='info'>No report history found for this user.</p>";
                return;
            }

            string html = "<table class='history-table'>";
            html += "<tr><th>ID</th><th>Date</th><th>Diagnosis</th><th>Action</th></tr>";

            foreach (var file in files)
            {
                try
                {
                    XElement reportXml = XElement.Load(file.FullName);

                    string reportId = reportXml.Attribute("ReportID")?.Value ?? "N/A";
                    string date = reportXml.Attribute("DateTime")?.Value ?? "N/A";
                    string diagnosis = reportXml.Element("Diagnosis")?.Value ?? "N/A";

                    // View button links to ShowReport.aspx with ?id=ReportID
                    string viewButton = $"<button class='btn-view' onclick=\"window.location='ShowReport.aspx?id={reportId}'\">View</button>";

                    html += $"<tr>" +
                            $"<td>{reportId}</td>" +
                            $"<td>{date}</td>" +
                            $"<td>{diagnosis}</td>" +
                            $"<td>{viewButton}</td>" +
                            $"</tr>";
                }
                catch
                {
                    // Skip invalid XML files
                }
            }

            html += "</table>";
            litHistory.Text = html;
        }
    }
}
