
import streamlit as st
import pandas as pd
import joblib
import os
from io import BytesIO
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Smart College Complaint Analyzer",
    page_icon="🎓",
    layout="wide"
)

# Load model and vectorizer
model = joblib.load("complaint_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page heading
st.title("🎓 Smart College Complaint Analyzer")
st.write("AI-based College Complaint Management System")
st.divider()

# Load complaint records
if os.path.exists("complaints_log.csv"):
    complaints_df = pd.read_csv("complaints_log.csv")

    if "status" not in complaints_df.columns:
        complaints_df["status"] = "Pending"
        complaints_df.to_csv("complaints_log.csv", index=False)
else:
    complaints_df = pd.DataFrame(
        columns=["complaint", "category", "status"]
    )

# Read model accuracy
accuracy = None

if os.path.exists("accuracy.txt"):
    with open("accuracy.txt", "r") as file:
        accuracy = float(file.read())

# Dashboard
st.header("📊 Admin Dashboard")

# Summary metrics
total = len(complaints_df)

pending = (
    complaints_df["status"].eq("Pending").sum()
    if total > 0 else 0
)

resolved = (
    complaints_df["status"].eq("Resolved").sum()
    if total > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Complaints", total)
col2.metric("Pending", int(pending))
col3.metric("Resolved", int(resolved))

if accuracy is not None:
    col4.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
else:
    col4.metric("Model Accuracy", "Not available")

st.divider()

if total > 0:

    # Category-wise charts
    st.subheader("Complaint Category Analysis")

    counts = complaints_df["category"].value_counts()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("Category-wise Bar Chart")
        st.bar_chart(counts)

    with chart_col2:
        st.write("Category-wise Pie Chart")

        fig, ax = plt.subplots()
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")

        st.pyplot(fig)
        plt.close(fig)

    # Most common category
    st.subheader("Most Common Complaint Category")
    st.success(counts.idxmax())

    st.divider()

    # Filter complaints
    st.subheader("🔎 Search and Filter Complaints")

    filter_col, search_col = st.columns(2)

    with filter_col:
        selected_status = st.selectbox(
            "Filter by Status",
            ["All", "Pending", "In Progress", "Resolved"]
        )

    with search_col:
        search_text = st.text_input(
            "Search by keyword",
            placeholder="Example: WiFi, bus, canteen"
        )

    filtered_df = complaints_df.copy()

    if selected_status != "All":
        filtered_df = filtered_df[
            filtered_df["status"] == selected_status
        ]

    if search_text:
        filtered_df = filtered_df[
            filtered_df["complaint"].str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

    st.write("Complaint Records")
    st.dataframe(filtered_df, use_container_width=True)

    # Excel download
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        filtered_df.to_excel(writer, index=False)

    st.download_button(
        label="📥 Download Complaints as Excel",
        data=output.getvalue(),
        file_name="college_complaints.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()

    # Complaint status chart
    st.subheader("Complaint Status Analysis")

    status_counts = complaints_df["status"].value_counts()
    st.bar_chart(status_counts)

    # Update status
    st.subheader("Update Complaint Status")

    selected_number = st.selectbox(
        "Select Complaint Number",
        list(range(1, total + 1))
    )

    new_status = st.selectbox(
        "Choose New Status",
        ["Pending", "In Progress", "Resolved"]
    )

    if st.button("Update Status"):
        selected_index = selected_number - 1

        complaints_df.loc[selected_index, "status"] = new_status

        complaints_df.to_csv(
            "complaints_log.csv",
            index=False
        )

        st.success("Complaint status updated successfully!")
        st.rerun()

else:
    st.info("No complaints submitted yet. Submit your first complaint below.")

st.divider()

# New complaint section
st.header("📝 Analyze a New Complaint")

complaint = st.text_area(
    "Enter your college complaint:",
    placeholder="Example: WiFi is not working in the computer lab."
)

if st.button("Analyze Complaint"):

    if complaint.strip() == "":
        st.warning("Please enter a complaint.")

    else:
        complaint_vector = vectorizer.transform([complaint])
        prediction = model.predict(complaint_vector)

        new_data = pd.DataFrame({
            "complaint": [complaint],
            "category": [prediction[0]],
            "status": ["Pending"]
        })

        new_data.to_csv(
            "complaints_log.csv",
            mode="a",
            header=not os.path.exists("complaints_log.csv"),
            index=False
        )

        st.success("Complaint analyzed and saved successfully!")

        st.subheader("Predicted Category")
        st.write(prediction[0])

        st.info("Complaint Status: Pending")
        st.rerun()
