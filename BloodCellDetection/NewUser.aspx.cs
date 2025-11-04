using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Xml;
using System.IO;
using Sysmgr;

namespace BloodCellDetection
{
    public partial class NewUser : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {

        }
        protected void btnRegister_Click(object sender, EventArgs e)
        {
            string name = txtName.Text.Trim();
            string phone = txtPhone.Text.Trim();
            string userId = txtUserId.Text.Trim();
            string password = txtPassword.Text.Trim();
            string confirmPassword = txtConfirmPassword.Text.Trim();

            // Validation
            if (string.IsNullOrEmpty(name) || string.IsNullOrEmpty(phone) ||
                string.IsNullOrEmpty(userId) || string.IsNullOrEmpty(password))
            {
                lblMessage.Text = "⚠ Please fill in all fields.";
                return;
            }

            if (password != confirmPassword)
            {
                lblMessage.Text = "⚠ Passwords do not match!";
                return;
            }

            // Path to Users.xml (inside App_Data folder)
            string filePath = Server.MapPath("~/App_Data/Users.xml");

            XmlDocument doc = new XmlDocument();

            // If file doesn't exist, create root <Users>
            if (!File.Exists(filePath))
            {
                XmlDeclaration xmlDecl = doc.CreateXmlDeclaration("1.0", "UTF-8", null);
                doc.AppendChild(xmlDecl);

                XmlElement root = doc.CreateElement("Users");
                doc.AppendChild(root);

                doc.Save(filePath);
            }

            doc.Load(filePath);

            // Check if UserID already exists
            XmlNode existingUser = doc.SelectSingleNode($"/Users/User[user_id='{userId}']");
            if (existingUser != null)
            {
                lblMessage.Text = "⚠ User ID already exists. Choose another.";
                return;
            }

            if (Validity.Check())
            {
                XmlElement newUser = doc.CreateElement("User");

                XmlElement xmlUserId = doc.CreateElement("user_id");
                xmlUserId.InnerText = userId;
                newUser.AppendChild(xmlUserId);

                XmlElement xmlName = doc.CreateElement("name");
                xmlName.InnerText = name;
                newUser.AppendChild(xmlName);

                XmlElement xmlPhone = doc.CreateElement("phone_no");
                xmlPhone.InnerText = phone;
                newUser.AppendChild(xmlPhone);

                XmlElement xmlPassword = doc.CreateElement("password");
                xmlPassword.InnerText = password;  // 🔒 You may later hash this
                newUser.AppendChild(xmlPassword);

                // Add to root <Users>
                doc.DocumentElement.AppendChild(newUser);
                doc.Save(filePath);

                lblMessage.ForeColor = System.Drawing.Color.Green;
                lblMessage.Text = "✅ User registered successfully!";
            }
            // Clear fields
            txtName.Text = txtPhone.Text = txtUserId.Text = txtPassword.Text = txtConfirmPassword.Text = "";
        }
    }
}