using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace BloodCellDetection
{
    public partial class HomeMaster : System.Web.UI.MasterPage
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            if (Session["UserName"] != null)
            {
                Label1.Text = "Welcome " + Session["UserName"];
            }
            else
            {
                Response.Redirect("Login.aspx");
            }
        }
    }
}