# Blood Cell Detection System

A comprehensive blood cell detection system that combines ASP.NET web application with Python machine learning backend for automated blood cell analysis and detection.

## 🩸 Project Overview

This system provides an automated solution for blood cell detection and analysis using computer vision and machine learning techniques. The application features a web-based interface built with ASP.NET and leverages Python-based machine learning models for accurate blood cell identification.

## 🏗️ Architecture

- **Frontend**: ASP.NET Web Application with HTML/CSS/JavaScript
- **Backend**: Python machine learning models
- **Models**: Keras/TensorFlow models for blood cell classification
- **Database**: XML-based user management

## 📁 Project Structure

```
├── BloodCellDetection/          # ASP.NET Web Application
│   ├── Dashboard.aspx           # Main dashboard
│   ├── Login.aspx              # User authentication
│   ├── NewUser.aspx            # User registration
│   ├── UHome.aspx              # User home page
│   ├── Python/                 # Python integration scripts
│   │   ├── Cell.py             # Cell detection logic
│   │   ├── Cell2.py            # Additional cell processing
│   │   ├── Detect.py           # Main detection script
│   │   └── Blood1.keras        # ML model file
│   └── App_Data/               # Application data
│       └── Users.xml           # User database
├── PythonCode/                 # Python ML models and scripts
│   ├── Blood.keras             # Primary ML model (121MB)
│   ├── Blood1.keras            # Secondary ML model (547MB)
│   └── Sample images/          # Test images
└── .gitignore                  # Git ignore rules
```

## 🔧 Features

- **User Authentication**: Secure login and registration system
- **Blood Cell Detection**: Automated detection using ML models
- **Dashboard Interface**: User-friendly web interface
- **Image Processing**: Real-time blood cell image analysis
- **Results Display**: Clear visualization of detection results

## 🚀 Getting Started

### Prerequisites

- .NET Framework
- Visual Studio or Visual Studio Code
- Python 3.x
- TensorFlow/Keras
- Required Python packages (see requirements)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shrey30/BloodCellDetection.git
   cd BloodCellDetection
   ```

2. Open the ASP.NET solution in Visual Studio:
   ```
   BloodCellDetection/BloodCellDetection.sln
   ```

3. Install Python dependencies:
   ```bash
   pip install tensorflow keras opencv-python numpy
   ```

4. Build and run the ASP.NET application

## 🧪 Usage

1. Start the ASP.NET web application
2. Register a new user or login with existing credentials
3. Upload blood cell images through the dashboard
4. View detection results and analysis

## 🤖 Machine Learning Models

The system uses two main Keras models:

- **Blood.keras** (121MB): Primary detection model
- **Blood1.keras** (547MB): Enhanced detection model

*Note: Model files are stored using Git LFS due to their large size.*

## 📊 Model Performance

The ML models are trained to detect and classify various types of blood cells with high accuracy. Detailed performance metrics and model architecture information can be found in the respective Python scripts.

## 🛠️ Technologies Used

- **ASP.NET Web Forms**: Frontend web application
- **C#**: Server-side logic
- **Python**: Machine learning backend
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Image processing
- **HTML/CSS/JavaScript**: Web interface
- **XML**: Data storage

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Shrey30**
- GitHub: [@shrey30](https://github.com/shrey30)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/shrey30/BloodCellDetection/issues).

## 📞 Support

If you have any questions or need help, please open an issue on GitHub.

---

⭐ **Star this repository if it helped you!** ⭐