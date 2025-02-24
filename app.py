import joblib
import pandas as pd

class AttritionPredictor:
    def __init__(self):
        """Initialize the predictor by loading the models and setting options."""
        self.rf_model = joblib.load("attrition_model_basic.pkl")  # Random Forest Model
        self.xgb_model = joblib.load("attrition_model_oversampled.pkl")  # XGBoost Model
        self.model = None
        self.model_name = ""

        # Department and Education Field options
        self.department_options = {
            1: "Human Resources",
            2: "Research & Development",
            3: "Sales"
        }
        self.education_field_options = {
            1: "Human Resources",
            2: "Life Sciences",
            3: "Marketing",
            4: "Medical",
            5: "Other",
            6: "Technical Degree"
        }

    def select_model(self):
        """Allow user to choose between Random Forest and XGBoost models."""
        while True:
            choice = input("\n🔹 Choose Model (RF for Random Forest / RFO for Random forest oversampled): ").strip().upper()
            if choice == "RF":
                self.model = self.rf_model
                self.model_name = "Random Forest"
                break
            elif choice == "RFO":
                self.model = self.xgb_model
                self.model_name = "Random Forest Oversampled"
                break
            else:
                print("❌ Invalid choice! Please enter 'RF' or 'XGB'.")

    def get_user_input(self):
        """Collect user input for employee details and return a processed DataFrame."""
        print("\n🔹 Enter Employee Details for Prediction 🔹\n")

        # Numeric Inputs
        age = int(input("🔹 Age: "))
        distance = int(input("🔹 Distance From Home: "))
        job_level = int(input("🔹 Job Level (1-5): "))
        monthly_income = int(input("🔹 Monthly Income: "))
        total_working_years = float(input("🔹 Total Working Years: "))  
        years_at_company = int(input("🔹 Years at Company: "))
        daily_rate = int(input("🔹 Daily Rate: "))

        # Select Department
        print("\n🔹 Select Department:")
        for key, value in self.department_options.items():
            print(f"{key}. {value}")
        dept_choice = int(input("Enter the number corresponding to the department: "))
        department = self.department_options.get(dept_choice, "Research & Development")  # Default if invalid input

        # Select Education Field
        print("\n🔹 Select Education Field:")
        for key, value in self.education_field_options.items():
            print(f"{key}. {value}")
        edu_choice = int(input("Enter the number corresponding to the education field: "))
        education_field = self.education_field_options.get(edu_choice, "Life Sciences")  # Default if invalid input

        # Education Level
        education = int(input("\n🔹 Education Level (1-5): "))

        # Create a dictionary for user input
        user_dict = {
            'Age': age,
            'DistanceFromHome': distance,
            'JobLevel': job_level,
            'MonthlyIncome': monthly_income,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'DailyRate': daily_rate,
            'Education': education
        }

        # One-Hot Encoding
        departments = ['Department_Human Resources', 'Department_Research & Development', 'Department_Sales']
        education_fields = ['Education_Human Resources', 'Education_Life Sciences', 'Education_Marketing', 
                            'Education_Medical', 'Education_Other', 'Education_Technical Degree']

        for dept in departments:
            user_dict[dept] = 1 if f"Department_{department}" == dept else 0

        for edu_field in education_fields:
            user_dict[edu_field] = 1 if f"Education_{education_field}" == edu_field else 0

        # Convert to DataFrame
        user_data = pd.DataFrame([user_dict])

        # Ensure correct column order
        expected_features = self.model.feature_names_in_
        user_data = user_data.reindex(columns=expected_features, fill_value=0)

        return user_data

    def predict(self):
        """Run the prediction using the selected model."""
        if not self.model:
            self.select_model()

        user_data = self.get_user_input()

        # Make prediction
        prediction = self.model.predict(user_data)[0]
        probability = self.model.predict_proba(user_data)[0][1]

        # Display results
        print("\n🔹 Prediction Result 🔹")
        print(f"📊 Model Used: {self.model_name}")
        if prediction == 1:
            print(f"⚠️ Employee is **likely to leave** the company. (Probability: {probability:.2f})")
        else:
            print(f"✅ Employee is **likely to stay**. (Probability: {probability:.2f})")

    def run(self):
        """Start the console application loop."""
        while True:
            self.predict()
            cont = input("\n🔄 Do you want to predict again? (yes/no): ").strip().lower()
            if cont != "yes":
                print("🚀 Exiting the app. Have a great day!")
                break

# Run the app
if __name__ == "__main__":
    app = AttritionPredictor()
    app.run()
