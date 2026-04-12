from shiny import App, ui, render, reactive
import pandas as pd
import joblib
import plotly.express as px
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=UserWarning)

# ====================== LOAD DATA & MODELS ======================
df = pd.read_csv("Preprocessed_Credit_Card_Dataset.csv")
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

    # === MAIN CONTENT WRAPPED IN ui.div() - THIS FIXES THE ERROR ===
    ui.div(
        ui.panel_conditional(
            "input.nav == 'overview'",
            ui.h3("📊 Dashboard Overview"),
            ui.layout_columns(
                ui.value_box("Total Records", f"{len(df):,}", theme="primary"),
                ui.value_box("Default Rate", f"{df['TARGET'].mean()*100:.2f}%", theme="danger"),
                ui.value_box("Avg Income", f"${df['INCOME'].mean():,.0f}", theme="success"),
            ),
            ui.card(ui.card_header("Sample Data"), ui.output_table("data_table"))
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

        ui.panel_conditional(
            "input.nav == 'prediction'",
            ui.h3("🔮 Credit Risk Prediction"),
            ui.card(
                ui.card_header("Select Model"),
                ui.input_radio_buttons("model_choice", None, 
                                      choices=["Logistic Regression", "XGBoost"], selected="XGBoost")
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
                    ui.input_select("car", "Owns Car?", ["Yes", "No"]),
                    ui.input_select("property", "Owns Property?", ["Yes", "No"]),
                    ui.input_select("education", "Education", 
                                   ["Higher education", "Secondary / secondary special"]),            
                )
            ),
            ui.input_action_button("predict_btn", "🚀 Predict Default Risk", 
                                  class_="btn btn-primary btn-lg w-100 mt-3"),
            ui.output_text_verbatim("prediction_result")
        )
    )
)

# ====================== SERVER ======================
def server(input, output, session):
    
    @output
    @render.table
    def data_table():
        return df.head(10)
    
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

    @reactive.Effect
    @reactive.event(input.predict_btn)
    def make_prediction():
        try:
        # ==================== COMPLETE INPUT DICTIONARY ====================
            input_data = pd.DataFrame([{
                'AGE': input.age(),
                'INCOME': input.income(),
                'YEARS_EMPLOYED': input.years_employed(),
                'FAMILY SIZE': input.family_size(),
                'GENDER': 1 if input.gender() == "Male" else 0,
                'CAR': 1 if input.car() == "Yes" else 0,
                'REALITY': 1 if input.property() == "Yes" else 0,
                'EDUCATION_TYPE': input.education(),
                'INCOME_TYPE': 'Working',                    # Default
                'NO_OF_CHILD': 1,                            # Default
                'BEGIN_MONTH': -10,                          # Default
                'FAMILY_TYPE': 'Married',                    # Default
                'HOUSE_TYPE': 'House / apartment',           # Default
                'FLAG_MOBIL': 1,                             # Default
                'WORK_PHONE': 0,                             # Default
                'PHONE': 0,                                  # Default
                'E_MAIL': 0,                                 # Default
                
                # Engineered Features (Very Important!)
                'INCOME_PER_FAMILY_MEMBER': input.income() / input.family_size(),
                'EMPLOYMENT_TO_AGE_RATIO': input.years_employed() / input.age(),
                'HAS_CHILD': 1 if input.family_size() > 1 else 0,
                'IS_SINGLE': 1 if input.family_size() == 1 else 0,
                'LOG_INCOME': np.log1p(input.income()),
                'AGE_GROUP': 'Middle_Age'                    # Default - you can make dynamic if needed
            }])

            # Select model and predict
            model = lr_model if input.model_choice() == "Logistic Regression" else xgb_model
            prob = model.predict_proba(input_data)[0][1]
            
            if prob > 0.5:
                ui.notification_show(f"⚠️ HIGH RISK - Likely to Default ({prob:.1%})", type="error", duration=10)
            else:
                ui.notification_show(f"✅ LOW RISK - Likely to Repay ({prob:.1%})", type="success", duration=10)
                
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")

# ====================== RUN APP ======================
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(port=8080)