from shiny import App, ui, render, reactive
import pandas as pd
import joblib
import plotly.express as px
import warnings
warnings.filterwarnings('ignore', category=UserWarning)



# ====================== LOAD DATA & MODELS ======================
df = pd.read_csv("Preprocessed_Credit_Card_Dataset.csv")
lr_model = joblib.load("logistic_regression_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

# ====================== UI ======================
app_ui = ui.page_sidebar(
    # Left Sidebar
    ui.sidebar(
        ui.p("🏠 Risk Dashboard", style="text-align: left; color: #6c757d; margin-top: -10px;"),
        ui.hr(),
        ui.input_radio_buttons(
            "nav",
            label=None,
            choices={
                "overview": "📊 Data Overview",
                "eda": "📈 Exploratory Data Analysis",
                "prediction": "🔮 Prediction"
            },
            selected="overview"
        ),
        width=290
    ),

    # Main Content
    ui.panel_conditional(
        "input.nav == 'overview'",
        ui.h3("Dataset Overview"),
        ui.value_box("Total Records", f"{len(df):,}", icon="database"),
        ui.value_box("Default Rate", f"{df['TARGET'].mean()*100:.2f}%", icon="exclamation-triangle", theme="danger"),
        ui.output_table("data_table")
    ),

    ui.panel_conditional(
        "input.nav == 'eda'",
        ui.h3("Exploratory Data Analysis"),
        ui.layout_columns(
            ui.card(
                ui.card_header("Target Distribution"),
                ui.output_plot("target_plot")
            ),
            ui.card(
                ui.card_header("Default Rate by Education"),
                ui.output_plot("education_plot")
            )
        )
    ),

    ui.panel_conditional(
        "input.nav == 'prediction'",
        ui.h3("Credit Risk Prediction"),
        
        ui.card(
            ui.card_header("Select Model"),
            ui.input_radio_buttons(
                "model_choice", 
                None,
                choices=["Logistic Regression", "XGBoost"],
                selected="XGBoost"
            )
        ),
        
        ui.layout_columns(
            ui.card(
                ui.input_numeric("age", "Age", value=35),
                ui.input_numeric("income", "Monthly Income", value=150000),
                ui.input_numeric("years_employed", "Years Employed", value=8),
                ui.input_numeric("family_size", "Family Size", value=3)
            ),
            ui.card(
                ui.input_select("gender", "Gender", ["Male", "Female"]),
                ui.input_select("car", "Owns Car", ["Yes", "No"]),
                ui.input_select("property", "Owns Property", ["Yes", "No"]),
                ui.input_select("education", "Education", 
                               ["Higher education", "Secondary / secondary special"])
            )
        ),
        
        ui.input_action_button("predict_btn", "🚀 Predict Default Risk", 
                              class_="btn-primary btn-lg w-100 mt-3"),
        ui.output_text_verbatim("prediction_result")
    ),

    title="💳 Credit Card Risk Predictor ", 
    style="color: #1f77b4; text-align: center;",
    id="main_nav"
)

# ====================== SERVER ======================
def server(input, output, session):
    
    @output
    @render.table
    def data_table():
        return df.head(10)
    
    @output
    @render.plot
    def target_plot():
        fig = px.pie(df, names='TARGET', title="Target Distribution")
        return fig
    
    @output
    @render.plot
    def education_plot():
        rate = df.groupby('EDUCATION_TYPE')['TARGET'].mean().reset_index()
        fig = px.bar(rate, x='EDUCATION_TYPE', y='TARGET', title="Default Rate by Education")
        return fig
    
    @reactive.Effect
    @reactive.event(input.predict_btn)
    def make_prediction():
        try:
            input_data = pd.DataFrame([{
                'AGE': input.age(),
                'INCOME': input.income(),
                'YEARS_EMPLOYED': input.years_employed(),
                'FAMILY SIZE': input.family_size(),
                'GENDER': 1 if input.gender() == "Male" else 0,
                'CAR': 1 if input.car() == "Yes" else 0,
                'REALITY': 1 if input.property() == "Yes" else 0,
                'EDUCATION_TYPE': input.education(),
                'INCOME_TYPE': 'Working',
                'NO_OF_CHILD': 1,
                'BEGIN_MONTH': -10,
                'FAMILY_TYPE': 'Married',
                'HOUSE_TYPE': 'House / apartment',
                'FLAG_MOBIL': 1,
                'WORK_PHONE': 0,
                'PHONE': 0,
                'E_MAIL': 0,
            }])
            
            model = lr_model if input.model_choice() == "Logistic Regression" else xgb_model
            prob = model.predict_proba(input_data)[0][1]
            
            if prob > 0.5:
                ui.notification_show(f"⚠️ HIGH RISK ({prob:.1%})", type="error")
            else:
                ui.notification_show(f"✅ LOW RISK ({prob:.1%})", type="success")
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")

# ====================== RUN APP ======================
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(port=8080)