import pandas as pd
from datetime import datetime
import math
import os
import random # For simulating ML and API responses

# --- Configuration (would typically come from a config file or database) ---
SETTINGS = {
    "DEFAULT_INTEREST_RATE": 10.0,
    "MIN_NET_MONTHLY_INCOME_LAND": 500,
    "MIN_NET_MONTHLY_INCOME_HOME": 1000,
    "MIN_NET_MONTHLY_INCOME_PERSONAL": 750,
    "BUSINESS_MIN_VINTAGE_YEARS": 2,
    "BUSINESS_MIN_MONTHLY_TURNOVER": 5000,
    "LTV_RATIO_LAND": 0.80, # Loan-to-Value for Land
    "LTV_RATIO_GOLD": 0.90, # Loan-to-Value for Gold
    "LTV_RATIO_HOME": 0.85, # Loan-to-Value for Home
    "DTI_THRESHOLD_LAND": 0.35,
    "DTI_THRESHOLD_EDUCATION": 0.40,
    "DTI_THRESHOLD_GOLD": 0.50, # Gold loans are more secured
    "DTI_THRESHOLD_HOME": 0.30,
    "DTI_THRESHOLD_BUSINESS": 0.40,
    "DTI_THRESHOLD_PERSONAL": 0.30,
    "BUSINESS_TURNOVER_TO_PROFIT_RATIO": 0.20, # 20% of turnover considered as profit for DTI
    "INTEREST_RATES_BASE": { # Base rates, ML model will adjust
        "EDUCATION": 8.5,
        "LAND": 9.0,
        "GOLD": 11.0,
        "HOME": 7.5,
        "BUSINESS": 10.0,
        "PERSONAL": 12.0
    }
}

# --- Global Data Storage (Conceptual Database ORM-like structure) ---
class CustomerDB:
    def __init__(self):
        self.customers_data = {} # Simulates a database table for customers

    def get_customer(self, identifier):
        # Could be CustomerID or Email
        if identifier in self.customers_data:
            return self.customers_data[identifier]
        for cust_id, cust_data in self.customers_data.items():
            if cust_data.get('Email', '').lower() == identifier.lower():
                return cust_data
        return None

    def add_customer(self, customer_data):
        if customer_data['CustomerID'] in self.customers_data:
            return False # CustomerID already exists
        self.customers_data[customer_data['CustomerID']] = customer_data
        return True

    def update_customer(self, customer_id, new_data):
        if customer_id in self.customers_data:
            self.customers_data[customer_id].update(new_data)
            return True
        return False

customer_db = CustomerDB()

class LoanApplicationDB:
    def __init__(self, filename='all_applications_log.csv'):
        self.filename = filename
        self.all_applications_log = pd.DataFrame(columns=[
            'Timestamp', 'CustomerID', 'Applicant_Type', 'Email', 'Full_Name', 'Phone_Num', 'DOB',
            'Street_Address', 'City', 'State', 'Zip_Code', 'KYC_Status', 'Net_Monthly_Income',
            'Existing_Monthly_EMI', 'Loan_Type', 'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount_Requested',
            'Loan_Amount_Term', 'Credit_History_Score', 'Property_Area', 'Land_Location_Area',
            'Estimated_Land_Collateral_Value', 'Business_Type', 'Business_Vintage',
            'Avg_Monthly_Turnover', 'Loan_Purpose', 'Other_Loan_Details', 'Eligibility_Status',
            'Reason_for_Ineligibility', 'Calculated_Interest_Rate', 'ML_Credit_Score', 'ML_Eligibility_Probability'
        ])
        if os.path.exists(self.filename):
            try:
                self.all_applications_log = pd.read_csv(self.filename)
            except Exception as e:
                print(f"Warning: Could not load existing log file {self.filename}: {e}. Starting fresh.")

    def log_application(self, details, status, reason=None, interest_rate=None, ml_credit_score=None, ml_eligibility_probability=None):
        log_entry = details.copy()
        log_entry['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry['Eligibility_Status'] = status
        log_entry['Reason_for_Ineligibility'] = reason
        log_entry['Calculated_Interest_Rate'] = interest_rate
        log_entry['ML_Credit_Score'] = ml_credit_score
        log_entry['ML_Eligibility_Probability'] = ml_eligibility_probability

        # Ensure all columns are present before concatenating
        for col in self.all_applications_log.columns:
            if col not in log_entry:
                log_entry[col] = None # Add missing columns with None

        self.all_applications_log = pd.concat([self.all_applications_log, pd.DataFrame([log_entry])], ignore_index=True)
        self.all_applications_log.to_csv(self.filename, index=False)

loan_application_db = LoanApplicationDB()

# --- Helper Functions ---
def get_valid_input(prompt, type_func=str, validation_func=None, error_message="Invalid input. Please try again."):
    while True:
        user_input = input(prompt).strip()
        if user_input.lower() == 'exit':
            return 'exit'
        try:
            value = type_func(user_input)
            if validation_func is None or validation_func(value):
                return value
            else:
                print(error_message)
        except ValueError:
            print(error_message)

def validate_date(date_str):
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def calculate_emi(principal, annual_interest_rate, loan_term_months):
    if annual_interest_rate == 0:
        return principal / loan_term_months
    monthly_interest_rate = (annual_interest_rate / 12) / 100
    if monthly_interest_rate == 0: # Handle very small rates or terms
        return principal / loan_term_months
    emi = principal * monthly_interest_rate * (1 + monthly_interest_rate)**loan_term_months / \
          ((1 + monthly_interest_rate)**loan_term_months - 1)
    return emi

# --- Simulated External Services (APIs) ---
class KYCService:
    @staticmethod
    def verify_customer_kyc(customer_id, email):
        # Simulate API call to an external KYC provider
        print(f"Simulating KYC verification for CustomerID: {customer_id}...")
        if customer_id.startswith("CUST") and random.random() > 0.1: # 90% chance of verified
            return "Verified", "KYC successful via external service."
        else:
            return "Pending", "KYC requires manual review or additional documentation."

class CreditBureauService:
    @staticmethod
    def get_credit_score(customer_id, full_name, dob):
        # Simulate API call to a credit bureau
        print(f"Simulating Credit Bureau check for CustomerID: {customer_id}...")
        # A more realistic simulation would use the provided details
        score = random.randint(300, 850) # FICO score range
        if score < 580:
            history = "Poor"
        elif score < 670:
            history = "Fair"
        elif score < 740:
            history = "Good"
        elif score < 800:
            history = "Very Good"
        else:
            history = "Excellent"
        return score, history, "Credit report fetched successfully."

class CollateralValuationService:
    @staticmethod
    def get_land_value(location, size_sqft=1000): # size_sqft is a placeholder
        print(f"Simulating Land Valuation for {location}...")
        # In a real system, this would use location data, property type, market trends etc.
        base_value = random.uniform(50000, 500000)
        if location.lower() == 'urban':
            return base_value * 1.5 + random.uniform(0, 50000)
        elif location.lower() == 'semiurban':
            return base_value * 1.0 + random.uniform(0, 20000)
        else: # Rural
            return base_value * 0.7 + random.uniform(0, 10000)

    @staticmethod
    def get_gold_value(weight_grams=100, purity_karats=22):
        print(f"Simulating Gold Valuation for {weight_grams}g, {purity_karats}K...")
        # Real-time gold prices would be fetched
        price_per_gram_24k = 65 # Example price per gram in USD
        value = (weight_grams * price_per_gram_24k * (purity_karats / 24)) * random.uniform(0.95, 1.05)
        return value

    @staticmethod
    def get_property_value(area, num_bedrooms=3, sq_footage=1500):
        print(f"Simulating Property Valuation for {area}...")
        base_value = random.uniform(150000, 1500000)
        if area.lower() == 'urban':
            return base_value * 1.8 + random.uniform(0, 100000)
        elif area.lower() == 'semiurban':
            return base_value * 1.2 + random.uniform(0, 50000)
        else: # Rural
            return base_value * 0.8 + random.uniform(0, 30000)

# --- Applicant Class (Object-Oriented Approach) ---
class Applicant:
    def __init__(self, customer_id=None, applicant_type='New', **kwargs):
        self.CustomerID = customer_id
        self.Applicant_Type = applicant_type
        self.Email = None
        self.Full_Name = None
        self.Phone_Num = None
        self.DOB = None
        self.Street_Address = None
        self.City = None
        self.State = None
        self.Zip_Code = None
        self.KYC_Status = 'Unverified'
        self.Net_Monthly_Income = 0.0
        self.Existing_Monthly_EMI = 0.0
        self.Credit_History_Score = None # New: Stores actual credit score
        self.Credit_History_Rating = None # New: Stores rating (Poor, Fair, Good, etc.)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        # Convert object attributes to a dictionary, useful for logging
        return {attr: getattr(self, attr) for attr in vars(self) if not attr.startswith('__')}

    def collect_new_customer_data(self):
        print("\n--- New Customer Registration ---")
        self.CustomerID = get_valid_input("Enter a new CustomerID (e.g., CUST001): ", str)
        if self.CustomerID == 'exit': return 'exit'
        if customer_db.get_customer(self.CustomerID):
            print(f"CustomerID '{self.CustomerID}' already exists. Please choose a different one.")
            return self.collect_new_customer_data()

        self.Email = get_valid_input("Enter your Email: ", str)
        if self.Email == 'exit': return 'exit'
        self.Full_Name = get_valid_input("Enter your Full Name: ", str)
        if self.Full_Name == 'exit': return 'exit'
        self.Phone_Num = get_valid_input("Enter your Phone Num: ", str)
        if self.Phone_Num == 'exit': return 'exit'
        self.DOB = get_valid_input("Enter your Date of Birth (YYYY-MM-DD): ", str, validate_date, "Invalid date format. Please use YYYY-MM-DD.")
        if self.DOB == 'exit': return 'exit'
        self.Street_Address = get_valid_input("Enter your Street Address: ", str)
        if self.Street_Address == 'exit': return 'exit'
        self.City = get_valid_input("Enter your City: ", str)
        if self.City == 'exit': return 'exit'
        self.State = get_valid_input("Enter your State: ", str)
        if self.State == 'exit': return 'exit'
        self.Zip_Code = get_valid_input("Enter your Zip Code: ", str)
        if self.Zip_Code == 'exit': return 'exit'
        
        # Initial KYC check (can be updated later)
        self.KYC_Status, kyc_message = KYCService.verify_customer_kyc(self.CustomerID, self.Email)
        print(f"KYC Status: {self.KYC_Status}. {kyc_message}")

        self.Net_Monthly_Income = get_valid_input("Enter your Net Monthly Income (after taxes, in USD): ", float, lambda x: x > 0, "Income must be positive.")
        if self.Net_Monthly_Income == 'exit': return 'exit'
        self.Existing_Monthly_EMI = get_valid_input("Enter your total Existing Monthly EMI (for all current loans, in USD): ", float, lambda x: x >= 0, "EMI cannot be negative.")
        if self.Existing_Monthly_EMI == 'exit': return 'exit'

        # Get credit score from external service
        self.Credit_History_Score, self.Credit_History_Rating, credit_message = CreditBureauService.get_credit_score(self.CustomerID, self.Full_Name, self.DOB)
        print(f"Credit Score: {self.Credit_History_Score} ({self.Credit_History_Rating}). {credit_message}")

        customer_db.add_customer(self.to_dict())
        return 'continue'

    @classmethod
    def load_or_create(cls):
        print("\n--- Applicant Identification ---")
        is_existing = get_valid_input("Are you an existing bank customer (yes/no)? ", str, lambda x: x.lower() in ['yes', 'no'], "Please enter 'yes' or 'no'.")
        if is_existing == 'exit': return 'exit', None

        if is_existing.lower() == 'yes':
            customer_id_or_email = get_valid_input("Please enter your CustomerID or Email: ", str)
            if customer_id_or_email == 'exit': return 'exit', None

            found_customer_data = customer_db.get_customer(customer_id_or_email)
            if found_customer_data:
                print(f"Welcome back, {found_customer_data['Full_Name']}!")
                applicant = cls(applicant_type='Existing', **found_customer_data)
                # Re-verify KYC and credit score for existing customers (real-time check)
                applicant.KYC_Status, kyc_message = KYCService.verify_customer_kyc(applicant.CustomerID, applicant.Email)
                print(f"Real-time KYC Status: {applicant.KYC_Status}. {kyc_message}")
                applicant.Credit_History_Score, applicant.Credit_History_Rating, credit_message = CreditBureauService.get_credit_score(applicant.CustomerID, applicant.Full_Name, applicant.DOB)
                print(f"Real-time Credit Score: {applicant.Credit_History_Score} ({applicant.Credit_History_Rating}). {credit_message}")
                return 'continue', applicant
            else:
                print("Customer not found in our records. Please register as a new customer.")
                applicant = cls(applicant_type='New')
                if applicant.collect_new_customer_data() == 'exit': return 'exit', None
                return 'continue', applicant
        else:
            applicant = cls(applicant_type='New')
            if applicant.collect_new_customer_data() == 'exit': return 'exit', None
            return 'continue', applicant

# --- Simulated Machine Learning Model for Eligibility and Risk ---
class MLLoanEligibilityModel:
    def __init__(self):
        # In a real scenario, this would load a pre-trained model (e.g., scikit-learn, TensorFlow)
        pass

    def predict_eligibility(self, applicant_data, loan_details, loan_type):
        # This is a highly simplified, rule-based simulation of an ML model.
        # A real ML model would use features like income, DTI, credit score, demographics,
        # loan amount, loan term, collateral value, etc., to make a prediction.

        features = {
            "Net_Monthly_Income": applicant_data.Net_Monthly_Income,
            "Existing_Monthly_EMI": applicant_data.Existing_Monthly_EMI,
            "Credit_History_Score": applicant_data.Credit_History_Score,
            "KYC_Status": applicant_data.KYC_Status,
            "Loan_Amount_Requested": loan_details.get('Loan_Amount_Requested', 0),
            "Loan_Amount_Term": loan_details.get('Loan_Amount_Term', 1),
            "Applicant_Income": loan_details.get('Applicant_Income', 0),
            "Coapplicant_Income": loan_details.get('Coapplicant_Income', 0),
            "Loan_Type": loan_type
        }

        # Add specific features for loan types
        if loan_type == "Land Loan":
            features["Estimated_Land_Collateral_Value"] = loan_details.get('Estimated_Land_Collateral_Value', 0)
        elif loan_type == "Gold Loan":
            features["Estimated_Gold_Collateral_Value"] = loan_details.get('Estimated_Gold_Collateral_Value', 0)
        elif loan_type == "Home Loan":
            features["Estimated_Property_Value"] = loan_details.get('Estimated_Property_Value', 0)
        elif loan_type == "Business Loan":
            features["Business_Vintage"] = loan_details.get('Business_Vintage', 0)
            features["Avg_Monthly_Turnover"] = loan_details.get('Avg_Monthly_Turnover', 0)

        # --- Simplified ML Logic ---
        credit_score_weight = 0.4
        income_weight = 0.3
        dti_weight = 0.2
        collateral_weight = 0.1 # For secured loans

        # Normalize Credit Score (e.g., 300-850 to 0-1)
        normalized_credit_score = (features["Credit_History_Score"] - 300) / 550 if features["Credit_History_Score"] else 0

        # Calculate DTI (simplified for ML model)
        total_income_for_dti_calc = features["Net_Monthly_Income"] + features["Applicant_Income"] + features["Coapplicant_Income"]
        if loan_type == "Business Loan":
            total_income_for_dti_calc += features.get("Avg_Monthly_Turnover", 0) * SETTINGS["BUSINESS_TURNOVER_TO_PROFIT_RATIO"]

        simulated_new_emi = calculate_emi(
            features["Loan_Amount_Requested"],
            SETTINGS["INTEREST_RATES_BASE"].get(loan_type.upper().replace(' ', '_'), SETTINGS["DEFAULT_INTEREST_RATE"]), # Use base rate for ML calc
            features["Loan_Amount_Term"]
        )
        total_monthly_payments = features["Existing_Monthly_EMI"] + simulated_new_emi
        
        dti = total_monthly_payments / total_income_for_dti_calc if total_income_for_dti_calc > 0 else 1.0 # High DTI if no income

        # Base eligibility probability
        eligibility_probability = 0.5

        # Adjust based on factors
        if normalized_credit_score > 0.7: eligibility_probability += 0.15
        elif normalized_credit_score < 0.3: eligibility_probability -= 0.20

        if dti < 0.25: eligibility_probability += 0.10
        elif dti > 0.45: eligibility_probability -= 0.15

        if features["Net_Monthly_Income"] < 500: eligibility_probability -= 0.10

        if features["KYC_Status"] != "Verified": eligibility_probability -= 0.30 # Major negative impact

        # Secured loans get a boost if collateral is strong
        if loan_type in ["Land Loan", "Gold Loan", "Home Loan"]:
            collateral_value = features.get("Estimated_Land_Collateral_Value") or features.get("Estimated_Gold_Collateral_Value") or features.get("Estimated_Property_Value")
            if collateral_value and features["Loan_Amount_Requested"] <= collateral_value * 0.7:
                eligibility_probability += 0.05
            elif features["Loan_Amount_Requested"] > collateral_value * 0.9: # Very high LTV
                eligibility_probability -= 0.10

        # Clamp probability between 0 and 1
        eligibility_probability = max(0.01, min(0.99, eligibility_probability))

        # Credit score for the ML model can be a simple average of key factors
        ml_credit_score = int(normalized_credit_score * 1000) # Scale it up for a "score"
        ml_credit_score += (1 - dti) * 100 # Lower DTI means higher score
        ml_credit_score = max(300, min(850, int(ml_credit_score)))


        return ml_credit_score, eligibility_probability

    def determine_interest_rate(self, base_rate, ml_credit_score, eligibility_probability):
        # Adjust interest rate based on ML model's perceived risk
        rate_adjustment = 0

        if ml_credit_score > 750:
            rate_adjustment = -1.5 # Lower rate for excellent credit
        elif ml_credit_score > 670:
            rate_adjustment = -0.5
        elif ml_credit_score < 580:
            rate_adjustment = 2.0 # Higher rate for poor credit
        elif ml_credit_score < 620:
            rate_adjustment = 1.0

        if eligibility_probability < 0.4:
            rate_adjustment += 1.0 # Higher risk, higher rate

        final_rate = max(4.0, base_rate + rate_adjustment) # Minimum rate 4%
        return final_rate

ml_model = MLLoanEligibilityModel()

# --- Loan Processor Class (Centralized Logic) ---
class LoanProcessor:
    def __init__(self, applicant: Applicant):
        self.applicant = applicant
        self.loan_details = {}
        self.loan_type_name = ""
        self.combined_details = {}
        self.ml_credit_score = None
        self.ml_eligibility_probability = None
        self.calculated_interest_rate = None

    def get_common_loan_details(self, loan_type_name):
        self.loan_type_name = loan_type_name
        print(f"\n--- Applying for: {self.loan_type_name} ---")
        print("Please provide the requested details:")
        
        self.loan_details['Gender'] = get_valid_input("Gender (Male/Female): ", str, lambda x: x.lower() in ['male', 'female'], "Please enter 'Male' or 'Female'.")
        if self.loan_details['Gender'] == 'exit': return 'exit'
        self.loan_details['Married'] = get_valid_input("Married (Yes/No): ", str, lambda x: x.lower() in ['yes', 'no'], "Please enter 'Yes' or 'No'.")
        if self.loan_details['Married'] == 'exit': return 'exit'
        self.loan_details['Dependents'] = get_valid_input("Number of Dependents (0, 1, 2, 3+): ", str, lambda x: x in ['0', '1', '2', '3+'], "Please enter 0, 1, 2, or 3+.")
        if self.loan_details['Dependents'] == 'exit': return 'exit'
        self.loan_details['Education'] = get_valid_input("Highest Education (Graduate/Not Graduate): ", str, lambda x: x.lower() in ['graduate', 'not graduate'], "Please enter 'Graduate' or 'Not Graduate'.")
        if self.loan_details['Education'] == 'exit': return 'exit'
        self.loan_details['Self_Employed'] = get_valid_input("Self Employed (Yes/No): ", str, lambda x: x.lower() in ['yes', 'no'], "Please enter 'Yes' or 'No'.")
        if self.loan_details['Self_Employed'] == 'exit': return 'exit'
        self.loan_details['Applicant_Income'] = get_valid_input("Applicant Income (in USD): ", float, lambda x: x >= 0, "Income must be non-negative.")
        if self.loan_details['Applicant_Income'] == 'exit': return 'exit'
        self.loan_details['Coapplicant_Income'] = get_valid_input("Co-applicant Income (in USD, enter 0 if none): ", float, lambda x: x >= 0, "Income must be non-negative.")
        if self.loan_details['Coapplicant_Income'] == 'exit': return 'exit'
        self.loan_details['Loan_Amount_Requested'] = get_valid_input("Loan Amount Requested (in USD): ", float, lambda x: x > 0, "Loan amount must be positive.")
        if self.loan_details['Loan_Amount_Requested'] == 'exit': return 'exit'
        self.loan_details['Loan_Amount_Term'] = get_valid_input("Loan Amount Term (in months, e.g., 180 for 15 years): ", int, lambda x: x > 0, "Loan term must be positive.")
        if self.loan_details['Loan_Amount_Term'] == 'exit': return 'exit'
        
        # Credit History is now derived from Credit Bureau Service in Applicant object
        self.loan_details['Credit_History_Score'] = self.applicant.Credit_History_Score
        self.loan_details['Credit_History_Rating'] = self.applicant.Credit_History_Rating
        
        return 'continue'

    def _apply_ml_and_rules(self, loan_type_enum):
        # Predict eligibility and get a refined credit score from ML model
        self.ml_credit_score, self.ml_eligibility_probability = ml_model.predict_eligibility(
            self.applicant, self.loan_details, self.loan_type_name
        )
        
        base_rate = SETTINGS["INTEREST_RATES_BASE"].get(loan_type_enum, SETTINGS["DEFAULT_INTEREST_RATE"])
        self.calculated_interest_rate = ml_model.determine_interest_rate(
            base_rate, self.ml_credit_score, self.ml_eligibility_probability
        )

        approved = True
        reasons = []

        # --- Rule-based checks (overlay on ML model for critical policy enforcement) ---
        # Rule 1: KYC Status (Critical)
        if self.applicant.KYC_Status.lower() != 'verified':
            approved = False
            reasons.append("KYC (Know Your Customer) status is not 'Verified'. Loans can only be approved for Verified customers.")

        # Rule 2: Credit Score Threshold (Critical)
        if self.applicant.Credit_History_Score < 580: # Example: Hard cutoff for "Poor" credit
            approved = False
            reasons.append(f"Credit Score ({self.applicant.Credit_History_Score}) is too low (Poor rating). Minimum score of 580 required.")

        # Calculate DTI for policy enforcement
        total_income = self.loan_details['Applicant_Income'] + self.loan_details['Coapplicant_Income']
        # Use applicant's net monthly income (after tax) for DTI, as it's the disposable income
        total_income_for_dti_check = self.applicant.Net_Monthly_Income

        if self.loan_type_name == "Business Loan":
             total_income_for_dti_check += (self.loan_details.get('Avg_Monthly_Turnover', 0) * SETTINGS["BUSINESS_TURNOVER_TO_PROFIT_RATIO"])

        if total_income_for_dti_check <= 0:
            approved = False
            reasons.append("No verifiable net monthly income. Repayment capacity cannot be assessed.")
            total_monthly_payments = 0 # Avoid division by zero
            dti_ratio = 1.0 # Max DTI
        else:
            new_loan_emi = calculate_emi(self.loan_details['Loan_Amount_Requested'], self.calculated_interest_rate, self.loan_details['Loan_Amount_Term'])
            total_monthly_payments = self.applicant.Existing_Monthly_EMI + new_loan_emi
            dti_ratio = total_monthly_payments / total_income_for_dti_check
            
            dti_threshold = SETTINGS[f"DTI_THRESHOLD_{loan_type_enum}"]
            if dti_ratio > dti_threshold:
                approved = False
                reasons.append(f"Your total estimated monthly loan payments ({total_monthly_payments:.2f} USD) would exceed {dti_threshold*100}% of your net monthly income ({dti_threshold * total_income_for_dti_check:.2f} USD). This indicates a high debt-to-income ratio, posing repayment risk.")

        # --- Loan Type Specific Rules ---
        if loan_type_enum == "LAND":
            collateral_value = self.loan_details.get('Estimated_Land_Collateral_Value', 0)
            if self.loan_details['Loan_Amount_Requested'] > SETTINGS["LTV_RATIO_LAND"] * collateral_value:
                approved = False
                reasons.append(f"Loan-to-Value (LTV) Ratio too high: Loan amount exceeds {SETTINGS['LTV_RATIO_LAND']*100}% of estimated collateral value.")
            if self.applicant.Net_Monthly_Income < SETTINGS["MIN_NET_MONTHLY_INCOME_LAND"]:
                approved = False
                reasons.append(f"Net Monthly Income is below the minimum required ({SETTINGS['MIN_NET_MONTHLY_INCOME_LAND']} USD) for a Land Loan.")

        elif loan_type_enum == "EDUCATION":
            if self.loan_details['Education'].lower() == 'not graduate' and self.applicant.Credit_History_Score < 650:
                approved = False
                reasons.append("Education Loan for 'Not Graduate' applicants requires a higher credit score or co-signer (not modeled here).")

        elif loan_type_enum == "GOLD":
            collateral_value = self.loan_details.get('Estimated_Gold_Collateral_Value', 0)
            if self.loan_details['Loan_Amount_Requested'] > SETTINGS["LTV_RATIO_GOLD"] * collateral_value:
                approved = False
                reasons.append(f"Loan-to-Value (LTV) Ratio too high: Loan amount exceeds {SETTINGS['LTV_RATIO_GOLD']*100}% of estimated gold collateral value.")
            if self.applicant.Net_Monthly_Income == 0 and self.loan_details['Loan_Amount_Requested'] > 0:
                 approved = False
                 reasons.append("No net monthly income declared. While secured by gold, a minimum repayment capacity is still expected.")

        elif loan_type_enum == "HOME":
            collateral_value = self.loan_details.get('Estimated_Property_Value', 0)
            if self.loan_details['Loan_Amount_Requested'] > SETTINGS["LTV_RATIO_HOME"] * collateral_value:
                approved = False
                reasons.append(f"Loan-to-Value (LTV) Ratio too high: Loan amount exceeds {SETTINGS['LTV_RATIO_HOME']*100}% of estimated property value.")
            if self.applicant.Net_Monthly_Income < SETTINGS["MIN_NET_MONTHLY_INCOME_HOME"]:
                approved = False
                reasons.append(f"Net Monthly Income is below the minimum required ({SETTINGS['MIN_NET_MONTHLY_INCOME_HOME']} USD) for a Home Loan.")

        elif loan_type_enum == "BUSINESS":
            if self.loan_details['Business_Vintage'] < SETTINGS["BUSINESS_MIN_VINTAGE_YEARS"]:
                approved = False
                reasons.append(f"Business Vintage too low: Minimum {SETTINGS['BUSINESS_MIN_VINTAGE_YEARS']} years in business generally required.")
            if self.loan_details['Avg_Monthly_Turnover'] < SETTINGS["BUSINESS_MIN_MONTHLY_TURNOVER"]:
                approved = False
                reasons.append(f"Average Monthly Business Turnover is below the minimum required ({SETTINGS['BUSINESS_MIN_MONTHLY_TURNOVER']} USD).")

        elif loan_type_enum == "PERSONAL":
            if self.applicant.Net_Monthly_Income < SETTINGS["MIN_NET_MONTHLY_INCOME_PERSONAL"]:
                approved = False
                reasons.append(f"Net Monthly Income is below the minimum required ({SETTINGS['MIN_NET_MONTHLY_INCOME_PERSONAL']} USD) for a Personal Loan.")
        
        # Final decision based on combined ML probability and hard rules
        if self.ml_eligibility_probability < 0.3: # If ML model is highly confident it's a bad risk
            approved = False
            reasons.append(f"Machine Learning model predicts low eligibility (Probability: {self.ml_eligibility_probability:.2f}).")
            
        if not approved and not reasons: # Fallback if approved=False but no specific reason was added
            reasons.append("Application denied based on internal policy evaluation and risk assessment.")

        return approved, reasons, new_loan_emi if 'new_loan_emi' in locals() else None

    def process_loan_application(self, loan_type_enum):
        self.combined_details = {**self.applicant.to_dict(), **self.loan_details, 'Loan_Type': self.loan_type_name}
        print("\nProcessing your application with advanced analytics...")

        approved, reasons, new_loan_emi = self._apply_ml_and_rules(loan_type_enum)

        print(f"ML Model Credit Score: {self.ml_credit_score}")
        print(f"ML Model Eligibility Probability: {self.ml_eligibility_probability:.2f}")

        if approved:
            print(f"\n--- Loan Eligibility Result for {self.loan_type_name} ---")
            print("Congratulations! Your Loan is APPROVED.")
            loan_application_db.log_application(self.combined_details, 'APPROVED', None,
                                                self.calculated_interest_rate,
                                                self.ml_credit_score,
                                                self.ml_eligibility_probability)
            if new_loan_emi:
                print(f"Your estimated monthly EMI for this loan will be: {new_loan_emi:.2f} USD")
            print(f"Your personalized annual interest rate: {self.calculated_interest_rate:.2f}%")
        else:
            print(f"\n--- Loan Eligibility Result for {self.loan_type_name} ---")
            print("We regret to inform you that your loan is NOT APPROVED at this time due to specific policy requirements.")
            print("Reason(s) for ineligibility:")
            for reason in reasons:
                print(f"- {reason}")
            loan_application_db.log_application(self.combined_details, 'NOT APPROVED', '; '.join(reasons),
                                                self.calculated_interest_rate,
                                                self.ml_credit_score,
                                                self.ml_eligibility_probability)
        print("Please contact your bank for more detailed information or to discuss alternatives.")
        print(f"Application details logged to {loan_application_db.filename}")
        print("----------------------------------------------------")
        return 'continue'

    def process_land_loan(self):
        if self.get_common_loan_details("Land Loan") == 'exit': return 'exit'
        self.loan_details['Land_Location_Area'] = get_valid_input("Land Location Area (Urban/Rural/Semiurban): ", str, lambda x: x.lower() in ['urban', 'rural', 'semiurban'], "Please enter Urban, Rural, or Semiurban.")
        if self.loan_details['Land_Location_Area'] == 'exit': return 'exit'
        
        # Use external valuation service
        self.loan_details['Estimated_Land_Collateral_Value'] = CollateralValuationService.get_land_value(self.loan_details['Land_Location_Area'])
        print(f"Estimated Land Collateral Value (via service): {self.loan_details['Estimated_Land_Collateral_Value']:.2f} USD")
        
        return self.process_loan_application("LAND")

    def process_education_loan(self):
        if self.get_common_loan_details("Education Loan") == 'exit': return 'exit'
        self.loan_details['Loan_Purpose'] = get_valid_input("Purpose of Education Loan (e.g., Tuition, Living Expenses, Books): ", str)
        if self.loan_details['Loan_Purpose'] == 'exit': return 'exit'
        return self.process_loan_application("EDUCATION")

    def process_gold_loan(self):
        if self.get_common_loan_details("Gold Loan") == 'exit': return 'exit'
        gold_weight = get_valid_input("Enter Gold Weight (in grams): ", float, lambda x: x > 0, "Weight must be positive.")
        if gold_weight == 'exit': return 'exit'
        gold_purity = get_valid_input("Enter Gold Purity (e.g., 22 for 22K): ", int, lambda x: 10 <= x <= 24, "Purity must be between 10 and 24 Karats.")
        if gold_purity == 'exit': return 'exit'
        
        # Use external valuation service
        self.loan_details['Estimated_Gold_Collateral_Value'] = CollateralValuationService.get_gold_value(gold_weight, gold_purity)
        print(f"Estimated Gold Collateral Value (via service): {self.loan_details['Estimated_Gold_Collateral_Value']:.2f} USD")
        return self.process_loan_application("GOLD")

    def process_home_loan(self):
        if self.get_common_loan_details("Home Loan") == 'exit': return 'exit'
        self.loan_details['Property_Area'] = get_valid_input("Property Area (Urban/Rural/Semiurban): ", str, lambda x: x.lower() in ['urban', 'rural', 'semiurban'], "Please enter Urban, Rural, or Semiurban.")
        if self.loan_details['Property_Area'] == 'exit': return 'exit'
        
        # Use external valuation service
        self.loan_details['Estimated_Property_Value'] = CollateralValuationService.get_property_value(self.loan_details['Property_Area'])
        print(f"Estimated Property Value (via service): {self.loan_details['Estimated_Property_Value']:.2f} USD")
        return self.process_loan_application("HOME")

    def process_business_loan(self):
        if self.get_common_loan_details("Business Loan") == 'exit': return 'exit'
        self.loan_details['Business_Type'] = get_valid_input("Business Type (e.g., Retail, Manufacturing, Service): ", str)
        if self.loan_details['Business_Type'] == 'exit': return 'exit'
        self.loan_details['Business_Vintage'] = get_valid_input("Business Vintage (Years in Business): ", int, lambda x: x >= 0, "Business vintage must be non-negative years.")
        if self.loan_details['Business_Vintage'] == 'exit': return 'exit'
        self.loan_details['Avg_Monthly_Turnover'] = get_valid_input("Average Monthly Business Turnover (in USD): ", float, lambda x: x >= 0, "Turnover must be non-negative.")
        if self.loan_details['Avg_Monthly_Turnover'] == 'exit': return 'exit'
        return self.process_loan_application("BUSINESS")

    def process_personal_loan(self):
        if self.get_common_loan_details("Personal Loan") == 'exit': return 'exit'
        self.loan_details['Loan_Purpose'] = get_valid_input("Purpose of Personal Loan (e.g., Wedding, Travel, Medical): ", str)
        if self.loan_details['Loan_Purpose'] == 'exit': return 'exit'
        self.loan_details['Other_Loan_Details'] = "N/A" # No specific additional details for now
        return self.process_loan_application("PERSONAL")


# --- Main Loan Application Flow ---
def main():
    print("--- Welcome to the Advanced Bank Loan Eligibility Checker ---")
    print("Leveraging Machine Learning and Simulated API Integrations.")
    
    while True:
        status, applicant = Applicant.load_or_create()
        if status == 'exit':
            print("Exiting application. Goodbye!")
            break
        if not applicant:
            print("Exiting application. Goodbye!")
            break

        loan_processor = LoanProcessor(applicant)

        print("\n--- Select Loan Type ---")
        print("1. Education Loan")
        print("2. Land Loan")
        print("3. Gold Loan")
        print("4. Home Loan")
        print("5. Business Loan")
        print("6. Personal Loan")
        loan_type_choice = get_valid_input("Enter your choice (1-6): ", int, lambda x: 1 <= x <= 6, "Invalid choice. Please enter a number between 1 and 6.")
        if loan_type_choice == 'exit':
            print("Exiting application. Goodbye!")
            break

        process_status = 'continue'
        if loan_type_choice == 1:
            process_status = loan_processor.process_education_loan()
        elif loan_type_choice == 2:
            process_status = loan_processor.process_land_loan()
        elif loan_type_choice == 3:
            process_status = loan_processor.process_gold_loan()
        elif loan_type_choice == 4:
            process_status = loan_processor.process_home_loan()
        elif loan_type_choice == 5:
            process_status = loan_processor.process_business_loan()
        elif loan_type_choice == 6:
            process_status = loan_processor.process_personal_loan()
        
        if process_status == 'exit':
            print("Exiting application. Goodbye!")
            break

        check_another = get_valid_input("Do you want to check another loan (yes/no)? ", str, lambda x: x.lower() in ['yes', 'no'], "Please enter 'yes' or 'no'.")
        if check_another == 'exit' or check_another.lower() == 'no':
            print("Thank you for using the Advanced Bank Loan Eligibility Checker. Goodbye!")
            break

if __name__ == "__main__":
    main()