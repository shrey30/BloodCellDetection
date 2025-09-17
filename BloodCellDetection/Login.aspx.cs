using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Xml;

namespace BloodCellDetection
{
    public partial class Login : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {

        }
        protected void btnLogin_Click(object sender, EventArgs e)
        {
            string userId = txtUserId.Text.Trim();
            string password = txtPassword.Text.Trim();

            string xmlPath = Server.MapPath("~/App_Data/Users.xml");

            XmlDocument doc = new XmlDocument();
            doc.Load(xmlPath);

            // Select user with matching user_id and password
            XmlNode userNode = doc.SelectSingleNode($"/Users/User[user_id='{userId}' and password='{password}']");

            if (userNode != null)
            {
                string name = userNode["name"].InnerText;
                Session["UserName"] = name;
                Response.Redirect("UHome.aspx");
            }
            else
            {
                lblMessage.Text = "Invalid User ID or Password.";
            }
        }
    }
}