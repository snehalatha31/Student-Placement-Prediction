from flask import Flask, render_template, request,redirect
import os
import sqlite3
import joblib
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
TEMPLATE_DIR = os.path.join(FRONTEND_DIR, "templates")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "placement.db")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "placement_model.pkl"
)


# --------------------------------------------------
# FLASK APPLICATION
# --------------------------------------------------

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)


# --------------------------------------------------
# MACHINE LEARNING MODEL
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# These MUST match the order used during model training
FEATURES = [
        "CGPA",
        "Tenth_Percentage",
        "Twelfth_Percentage",
        "Backlogs",
        "Attendance",
        "Internship",
        "Projects",
        "Communication_Skill",
        "Technical_Skill"
]



# --------------------------------------------------
# CREATE DATABASE
# --------------------------------------------------

def create_database():

    os.makedirs(DATABASE_DIR, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            cgpa REAL,
            tenth REAL,
            twelfth REAL,
            backlogs INTEGER,
            attendance REAL,
            internship INTEGER,
            projects INTEGER,
            communication REAL,
            technical REAL,

            prediction INTEGER,
            result TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


# Create database when application starts
create_database()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------------
        # Get data from frontend
        # ------------------------------------------

        cgpa = float(request.form["cgpa"])

        tenth = float(request.form["tenth"])

        twelfth = float(request.form["twelfth"])

        backlogs = int(request.form["backlogs"])

        attendance = float(request.form["attendance"])

        internship = int(request.form["internship"])

        projects = int(request.form["projects"])

        communication = float(
            request.form["communication"]
        )

        technical = float(
            request.form["technical"]
        )


        # ------------------------------------------
        # Prepare input for ML model
        # ------------------------------------------

        input_data = pd.DataFrame(
            [[
                cgpa,
                tenth,
                twelfth,
                backlogs,
                attendance,
                internship,
                projects,
                communication,
                technical
            ]],
            columns=FEATURES
        )


        # ------------------------------------------
        # Make prediction
        # ------------------------------------------

        prediction = int(
            model.predict(input_data)[0]
        )


        # ------------------------------------------
        # Convert prediction to text
        # ------------------------------------------

        if prediction == 1:

            result = "Eligible for Placement"

        else:

            result = "Not Eligible for Placement"


        # ------------------------------------------
        # Calculate confidence
        # ------------------------------------------

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )


        # ------------------------------------------
        # SAVE DATA TO DATABASE
        # ------------------------------------------

        connection = sqlite3.connect(
            DATABASE_PATH
        )

        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO students (

                cgpa,
                tenth,
                twelfth,
                backlogs,
                attendance,
                internship,
                projects,
                communication,
                technical,
                prediction,
                result

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            cgpa,
            tenth,
            twelfth,
            backlogs,
            attendance,
            internship,
            projects,
            communication,
            technical,
            prediction,
            result
        ))

        connection.commit()

        connection.close()


        # ------------------------------------------
        # SEND RESULT TO FRONTEND
        # ------------------------------------------

        return render_template(

            "result.html",

            result=result,

            confidence=confidence,

            cgpa=cgpa,

            tenth=tenth,

            twelfth=twelfth,

            backlogs=backlogs,

            attendance=attendance,

            internship="Yes" if internship == 1 else "No",

            projects=projects,

            communication=communication,

            technical=technical
        )


    except Exception as error:

        return render_template(

            "result.html",

            error=str(error)
        )


# --------------------------------------------------
# VIEW STORED STUDENTS
# --------------------------------------------------

@app.route("/students")
def students():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM students
        ORDER BY id DESC
    """)

    students_data = cursor.fetchall()

    connection.close()

    return render_template(
        "students.html",
        students=students_data
    )

# --------------------------------------------------
# UPDATE STUDENT - SHOW UPDATE PAGE
# --------------------------------------------------

@app.route("/update/<int:student_id>", methods=["GET"])
def update_student(student_id):

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,)
    )

    student = cursor.fetchone()

    connection.close()

    if student is None:

        return "Student not found"

    return render_template(
        "update.html",
        student=student
    )


# --------------------------------------------------
# UPDATE STUDENT - SAVE CHANGES
# --------------------------------------------------

@app.route("/update/<int:student_id>", methods=["POST"])
def save_updated_student(student_id):

    try:

        # Get updated values from update.html

        cgpa = float(request.form["cgpa"])

        tenth = float(request.form["tenth"])

        twelfth = float(request.form["twelfth"])

        backlogs = int(request.form["backlogs"])

        attendance = float(request.form["attendance"])

        internship = int(request.form["internship"])

        projects = int(request.form["projects"])

        communication = float(
            request.form["communication"]
        )

        technical = float(
            request.form["technical"]
        )


        # ------------------------------------------
        # Prepare new data for ML model
        # ------------------------------------------

        input_data = pd.DataFrame(
            [[
                cgpa,
                tenth,
                twelfth,
                backlogs,
                attendance,
                internship,
                projects,
                communication,
                technical
            ]],
            columns=FEATURES
        )


        # ------------------------------------------
        # Predict again
        # ------------------------------------------

        prediction = int(
            model.predict(input_data)[0]
        )


        # ------------------------------------------
        # Convert prediction to text
        # ------------------------------------------

        if prediction == 1:

            result = "Eligible for Placement"

        else:

            result = "Not Eligible for Placement"


        # ------------------------------------------
        # Update database
        # ------------------------------------------

        connection = sqlite3.connect(
            DATABASE_PATH
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE students

            SET
                cgpa = ?,
                tenth = ?,
                twelfth = ?,
                backlogs = ?,
                attendance = ?,
                internship = ?,
                projects = ?,
                communication = ?,
                technical = ?,
                prediction = ?,
                result = ?

            WHERE id = ?
            """,

            (
                cgpa,
                tenth,
                twelfth,
                backlogs,
                attendance,
                internship,
                projects,
                communication,
                technical,
                prediction,
                result,
                student_id
            )
        )

        connection.commit()

        connection.close()


        # Go back to stored students page

        return redirect("/students")


    except Exception as error:

        return f"Error updating student: {error}"


# --------------------------------------------------
# DELETE STUDENT
# --------------------------------------------------

@app.route("/delete/<int:student_id>", methods=["POST"])
def delete_student(student_id):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM students WHERE id = ?",
        (student_id,)
    )

    connection.commit()

    connection.close()

    return redirect("/students")
# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)