from shiny import App, ui, render, reactive
import pandas as pd
import joblib
import plotly.express as px
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=UserWarning)

# ====================== LOAD DATA & MODELS ======================
df = pd.read_csv("Preprocessed_Credit_Card_Dataset.csv")
gui_test_df = pd.read_csv("GUI_Test_Data_Set.csv")

lr_model = joblib.load("logistic_regression_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

# ====================== UI ======================

app_ui = ui.page_sidebar(
    # Left Sidebar
    ui.sidebar(
        ui.h1("💳 Credit Card Fraud Detection", style="color: #1f77b4; text-align: center;"),
        ui.hr(),
        ui.input_radio_buttons(
            "nav",
            label=None,
            choices={
                "overview": "📊 Data Overview",
                "eda": "📈 Exploratory Data Analysis",
                "prediction": "🔮 Risk Prediction"
            },
            selected="overview"
        ),
        width=290
    ),

    ui.div(
            # ==================== DATA OVERVIEW TAB ====================
        ui.panel_conditional(
            "input.nav == 'overview'",
            ui.h3("📊 Dataset Overview"),
            
            ui.card(
                ui.card_header(ui.h5("📋 Credit Card Fraud Detection Dataset"), style="text-align: center;"),
                ui.card_body(
                    ui.markdown("""
                        **This dataset contains 25,134 credit card transactions** collected over **60 months**.

                        It includes **19 features** (12 numerical + 7 categorical) and is used to detect fraudulent transactions in real-world scenarios.
                    """),
                    
                    ui.layout_columns(
                        ui.value_box(
                            "Total Transactions", 
                            "25,134", 
                            theme="primary",
                            showcase=ui.HTML("📊")
                        ),
                        ui.value_box(
                            "Fraudulent Cases", 
                            "420", 
                            theme="danger",
                            showcase=ui.HTML("⚠️")
                        ),
                        ui.value_box(
                            "Fraud Rate", 
                            "1.6%", 
                            theme="warning",
                            showcase=ui.HTML("📉")
                        ),
                        ui.value_box(
                            "Time Period", 
                            "60 Months", 
                            theme="success",
                            showcase=ui.HTML("📅")
                        )
                    ),
                    
                    ui.hr(),
                    
                    ui.markdown("""
                        ### Feature Information
                        - **Target Variable**: `TARGET` (Binary)
                        - **1** = Fraudulent Transaction  
                        - **0** = Normal Transaction
                        
                        - **Data Types**: 12 Numerical | 7 Categorical
                        
                        - **Missing Values**: Present in some categorical & nominal variables
                    """)
                ),
                style="margin-bottom: 20px;"
    
            ),

            # Full GUI Test Set
            ui.h5(f"GUI Test Set ({len(gui_test_df):,} rows)"),
            ui.card(
                ui.card_header("GUI Test Set Preview"),
                ui.output_table("gui_test_table", height="400px"),
                style="""
                    max-height: 500px; 
                    overflow-y: auto; 
                    border: 1px solid #dee2e6;
                """
            )
        ),

        ui.panel_conditional(
            "input.nav == 'eda'",
            ui.h3("📈 Exploratory Data Analysis"),
            ui.navset_tab(
                ui.nav_panel("Target Overview",
                    ui.layout_columns(
                        ui.card(ui.card_header("Pie Chart"), ui.output_plot("target_pie")),
                        ui.card(ui.card_header("Bar Chart"), ui.output_plot("target_bar"))
                    )
                ),
                ui.nav_panel("Demographics",
                    ui.layout_columns(
                        ui.card(ui.card_header("Age Distribution"), ui.output_plot("age_hist")),
                        ui.card(ui.card_header("Default Rate by Gender"), ui.output_plot("gender_bar"))
                    )
                ),
                ui.nav_panel("Financial Factors",
                    ui.layout_columns(
                        ui.card(ui.card_header("Default Rate by Education"), ui.output_plot("education_bar")),
                        ui.card(ui.card_header("Default Rate by Income Type"), ui.output_plot("income_type_bar"))
                    )
                )
            )
        ),

        # ====================== PREDICTION TAB ======================
        ui.panel_conditional(
            "input.nav == 'prediction'",
            ui.h3("🔍 Credit Card Fraud Detection"),
            
            ui.card(
                ui.card_header("Select Model"),
                ui.input_radio_buttons("model_choice", None, 
                                    choices=["Logistic Regression", "XGBoost"], 
                                    selected="XGBoost")
            ),
            
            ui.layout_columns(
                ui.card(
                    ui.input_numeric("age", "Age", value=35, min=18, max=100),
                    ui.input_numeric("income", "Monthly Income ($)", value=150000, min=0),
                    ui.input_numeric("years_employed", "Years Employed", value=8, min=0),
                    ui.input_numeric("family_size", "Family Size", value=3, min=1),
                    ui.input_numeric("begin_month", "Begin Month (e.g. 10)", value=30),
                    
                    # Dynamic Age Group Display
                    ui.output_text("dynamic_age_group")
                ),
                
                ui.card(
                    ui.input_select("gender", "Gender", ["Male", "Female"]),
                    ui.input_select("car", "Owns Car?", ["Yes", "No"]),
                    ui.input_select("property", "Owns Property?", ["Yes", "No"]),
                    ui.input_select("education", "Education", 
                                ["Higher education", "Secondary / secondary special", 
                                    "Incomplete higher", "Lower secondary", "Academic degree"]),
                    ui.input_select("income_type", "Income Type", 
                                ["Working", "Commercial associate", "State servant", "Pensioner", "Student"]),
                    ui.input_select("family_type", "Family Type", 
                                ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]),
                    ui.input_select("house_type", "House Type", 
                                ["House / apartment", "With parents", "Municipal apartment", 
                                    "Rented apartment", "Office apartment", "Co-op apartment"]),
                )
            ),
            
            ui.layout_columns(
                ui.card(
                    ui.h6("Contact Flags"),
                    ui.input_switch("flag_mobil", "Has Mobile", value=True),
                    ui.input_switch("work_phone", "Work Phone", value=False),
                    ui.input_switch("phone", "Phone", value=False),
                    ui.input_switch("email", "Email", value=False),
                )
            ),
            
            ui.input_action_button("predict_btn", "🔍 Check Transaction", 
                                class_="btn btn-primary btn-lg w-100 mt-3"),
            
            ui.output_text_verbatim("prediction_result", placeholder="Prediction will appear here...")
        )
    )
)

# ====================== SERVER ======================
def server(input, output, session):

    @output
    @render.table
    def gui_test_table():
        return gui_test_df   
    
    @output
    @render.plot
    def target_pie():
        fig = px.pie(df, names='TARGET', hole=0.4, title="Target Distribution")
        return fig

    @output
    @render.plot
    def target_bar():
        fig = px.histogram(df, x='TARGET', title="Target Distribution (Bar)")
        return fig

    @output
    @render.plot
    def age_hist():
        fig = px.histogram(df, x='AGE', color='TARGET', nbins=30, title="Age Distribution")
        return fig

    @output
    @render.plot
    def gender_bar():
        rate = df.groupby('GENDER')['TARGET'].mean().reset_index()
        fig = px.bar(rate, x='GENDER', y='TARGET', title="Default Rate by Gender")
        return fig

    @output
    @render.plot
    def education_bar():
        rate = df.groupby('EDUCATION_TYPE')['TARGET'].mean().reset_index()
        fig = px.bar(rate, x='EDUCATION_TYPE', y='TARGET', title="Default Rate by Education")
        return fig

    @output
    @render.plot
    def income_type_bar():
        rate = df.groupby('INCOME_TYPE')['TARGET'].mean().reset_index()
        fig = px.bar(rate, x='INCOME_TYPE', y='TARGET', title="Default Rate by Income Type")
        return fig

    @output
    @render.text
    def dynamic_age_group():
        age = input.age()
        if age <= 25:
            group = "Young"
        elif age <= 35:
            group = "Young_Adult"
        elif age <= 45:
            group = "Middle_Age"
        elif age <= 55:
            group = "Senior"
        else:
            group = "Elderly"
        
        return f"**Detected Age Group:** {group}"


    @reactive.Effect
    @reactive.event(input.predict_btn)
    def make_prediction():
        try:
            age = input.age()
            
            # Dynamic Age Group
            if age <= 25:
                age_group = "Young"
            elif age <= 35:
                age_group = "Young_Adult"
            elif age <= 45:
                age_group = "Middle_Age"
            elif age <= 55:
                age_group = "Senior"
            else:
                age_group = "Elderly"

            # Prepare Input Data
            input_dict = {
                'AGE': age,
                'INCOME': input.income(),
                'YEARS_EMPLOYED': input.years_employed(),
                'FAMILY SIZE': input.family_size(),
                'BEGIN_MONTH': input.begin_month(),
                'GENDER': 1 if input.gender() == "Male" else 0,
                'CAR': 1 if input.car() == "Yes" else 0,
                'REALITY': 1 if input.property() == "Yes" else 0,
                'EDUCATION_TYPE': input.education(),
                'INCOME_TYPE': input.income_type(),
                'FAMILY_TYPE': input.family_type(),
                'HOUSE_TYPE': input.house_type(),
                'FLAG_MOBIL': 1 if input.flag_mobil() else 0,
                'WORK_PHONE': 1 if input.work_phone() else 0,
                'PHONE': 1 if input.phone() else 0,
                'E_MAIL': 1 if input.email() else 0,
                
                'NO_OF_CHILD': max(0, input.family_size() - 1),
                'INCOME_PER_FAMILY_MEMBER': round(input.income() / input.family_size(), 2),
                'EMPLOYMENT_TO_AGE_RATIO': round(input.years_employed() / age, 4) if age > 0 else 0,
                'HAS_CHILD': 1 if input.family_size() > 1 else 0,
                'IS_SINGLE': 1 if input.family_type() in ["Single / not married", "Separated", "Widow"] else 0,
                'LOG_INCOME': np.log1p(input.income()),
                'AGE_GROUP': age_group
            }

            input_data = pd.DataFrame([input_dict])

            # Get Predictions from BOTH Models
            lr_prob = lr_model.predict_proba(input_data)[0][1]
            xgb_prob = xgb_model.predict_proba(input_data)[0][1]

            lr_pred = 1 if lr_prob > 0.5 else 0
            xgb_pred = 1 if xgb_prob > 0.5 else 0

            # ==================== RESULT DISPLAY ====================
            result_text = f"""
    🔍 CREDIT CARD FRAUD PREDICTION RESULTS

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Logistic Regression Model:
    → {'🚨 FRAUD' if lr_pred == 1 else '✅ NON-FRAUD'}
    → Probability : {lr_prob:.1%}

    XGBoost Model:
    → {'🚨 FRAUD' if xgb_pred == 1 else '✅ NON-FRAUD'}
    → Probability : {xgb_prob:.1%}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Age Group    : {age_group}
    Begin Month  : {input.begin_month()}
            """

            if lr_pred != xgb_pred:
                result_text += "\n⚠️ ALERT: Both models gave DIFFERENT predictions!"

            ui.update_text("prediction_result", value=result_text)

            # Notification based on selected model in radio button
            if input.model_choice() == "Logistic Regression":
                final_prob = lr_prob
            else:
                final_prob = xgb_prob

            msg = "🚨 FRAUD DETECTED" if final_prob > 0.5 else "✅ NON-FRAUD"
            ui.notification_show(msg, 
                                type="error" if final_prob > 0.5 else "success", 
                                duration=10)

        except Exception as e:
            ui.update_text("prediction_result", value=f"❌ Error: {str(e)}")
            ui.notification_show(f"Error: {str(e)}", type="error", duration=12)


# ====================== RUN APP ======================
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(port=8080)