using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace BloodCellDetection
{
    public partial class Home : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            lblMessage.Text = "";
        }

        protected void Button1_Click(object sender, EventArgs e)
        {
            if (FileUpload1.HasFile)
            {
                string extension = Path.GetExtension(FileUpload1.FileName).ToLower();
                string[] allowedExtensions = { ".jpg", ".jpeg", ".png", ".gif", ".bmp" };

                if (allowedExtensions.Contains(extension))
                {
                    try
                    {
                        string filename = Path.GetFileName(FileUpload1.FileName);
                        string savePath = Server.MapPath("~/Python/test.jpg");
                        FileUpload1.SaveAs(savePath);
                        Image1.ImageUrl = "~/Python/test.jpg";
                        lblMessage.ForeColor = System.Drawing.Color.Green;
                        lblMessage.Text = "Image uploaded successfully: " + filename;
                        TextWriter wr = File.CreateText(Server.MapPath("~/Python/tt.txt"));
                        wr.WriteLine("Hello World");
                        wr.Close();
                        wr = File.CreateText(Server.MapPath("~/Python/demo1.txt"));
                        wr.WriteLine("Hello World");
                        wr.Close();
                    }
                    catch (Exception ex)
                    {
                        lblMessage.ForeColor = System.Drawing.Color.Red;
                        lblMessage.Text = "Upload failed: " + ex.Message;
                    }
                }
                else
                {
                    lblMessage.ForeColor = System.Drawing.Color.Red;
                    lblMessage.Text = "Only image files (.jpg, .png, .gif, .bmp) are allowed.";
                }
            }
            else
            {
                lblMessage.ForeColor = System.Drawing.Color.Red;
                lblMessage.Text = "Please select a file to upload.";
            }
        }
    }
}