from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("house.pkl")

@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        age = int(request.form["age"])

        prediction = model.predict([[area,bedrooms,age]])

        return render_template("index.html", result=prediction[0])

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)