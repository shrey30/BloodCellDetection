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
    public partial class GenerateReport : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            if (Session["UserName"] == null)
                Session["UserName"] = "JohnDoe";  // example username

            string username = Session["UserName"].ToString();
            string reportId = Guid.NewGuid().ToString();
            DateTime reportDate = DateTime.Now;

            Random rnd = new Random();

            // Step 1️⃣: Random test results (with numeric ranges)
            List<BloodTest> tests = new List<BloodTest>
        {
            new BloodTest {
                TestName = "RBC",
                NumericResult = Math.Round(4.0 + rnd.NextDouble() * 2.0, 2),
                MinRange = 4.5, MaxRange = 5.9,
                NormalRange = "4.5 - 5.9 million/µL"
            },
            new BloodTest {
                TestName = "WBC",
                NumericResult = rnd.Next(4000, 11000),
                MinRange = 4000, MaxRange = 11000,
                NormalRange = "4,000 - 11,000 /µL"
            },
            new BloodTest {
                TestName = "Platelet",
                NumericResult = rnd.Next(150000, 450000),
                MinRange = 150000, MaxRange = 450000,
                NormalRange = "150,000 - 450,000 /µL"
            },
            new BloodTest {
                TestName = "Hemoglobin",
                NumericResult = Math.Round(12.0 + rnd.NextDouble() * 6.0, 1),
                MinRange = 13.5, MaxRange = 17.5,
                NormalRange = "13.5 - 17.5 g/dL"
            }
        };

            // Step 2️⃣: Prepare display results (add units)
            foreach (var t in tests)
            {
                switch (t.TestName)
                {
                    case "RBC":
                        t.DisplayResult = $"{t.NumericResult} million/µL";
                        break;
                    case "WBC":
                        t.DisplayResult = $"{t.NumericResult:N0} /µL";
                        break;
                    case "Platelet":
                        t.DisplayResult = $"{t.NumericResult:N0} /µL";
                        break;
                    case "Hemoglobin":
                        t.DisplayResult = $"{t.NumericResult} g/dL";
                        break;
                }
            }

            // Step 3️⃣: Auto-generate Diagnosis
            List<string> abnormalities = new List<string>();

            foreach (var test in tests)
            {
                if (test.NumericResult < test.MinRange)
                    abnormalities.Add($"Low {test.TestName}");
                else if (test.NumericResult > test.MaxRange)
                    abnormalities.Add($"High {test.TestName}");
            }

            string diagnosis;
            if (abnormalities.Count == 0)
            {
                diagnosis = "Normal Blood cell Count. No abnormalities detected.";
            }
            else
            {
                diagnosis = "Abnormal Blood Count detected: " + string.Join(", ", abnormalities) + ".";
            }

            // Step 4️⃣: Build XML Report
            XElement reportXml = new XElement("BloodReport",
                new XAttribute("ReportID", reportId),
                new XAttribute("UserName", username),
                new XAttribute("DateTime", reportDate.ToString("yyyy-MM-dd HH:mm:ss")),
                new XElement("Tests",
                    from test in tests
                    select new XElement("Test",
                        new XElement("TestName", test.TestName),
                        new XElement("Result", test.DisplayResult),
                        new XElement("NormalRange", test.NormalRange)
                    )
                ),
                new XElement("Diagnosis", diagnosis)
            );

            // Step 5️⃣: Save to XML file
            string folderPath = Server.MapPath("~/App_Data/Reports/");
            if (!Directory.Exists(folderPath))
                Directory.CreateDirectory(folderPath);

            string fileName = $"{username}_{reportId}.xml";
            string filePath = Path.Combine(folderPath, fileName);
            reportXml.Save(filePath);

            // Step 6️⃣: Show report on screen
            //Response.ContentType = "text/xml";
            //Response.Write(reportXml.ToString());
            Response.Redirect("Report.aspx");
        }
    }
}