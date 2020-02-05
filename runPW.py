from flask import Flask, jsonify, request
import json
import numpy as np
import pickle


with open("modelPW.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def index():
    return \
    """
    <!DOCTYPE html>
    <html>
    <body>
    
    <h2>Predict Death Probability in Five Years</h2>
    
    <form method="POST" action="/result">
      Age:<br>
      <input type="number" name="age" value=50>
      <br><br>
      Year of Operation:<br>
      19<input type="number" name="year" value=60>
      <br><br>
      Number of Axillary Nodes:<br>
      <input type="number" name="num_Axi" value=3>
      <br><br>
      <input type="submit" value="Submit">
    </form> 
    </body>
    </html>
    """


@app.route('/result', methods=['POST'])
def result():
    PROPERTY_TYPE = request.form["PROPERTY_TYPE"]
    CITY = request.form["CITY"]
    BEDS = request.form["BEDS"]
    BATHS = request.form["BATHS"]
    #LOCATION = request.form["LOCATION"]
    SQUARE_FEET = request.form["SQUARE_FEET"]
    LOT_SIZE = request.form["LOT_SIZE"]
    YEAR_BUILT = request.form["YEAR_BUILT"]
    DAYS_ON_MARKET = request.form["DAYS_ON_MARKET"]
    HOAperMONTH = request.form["HOAperMONTH"]
    LATITUDE = request.form["LATITUDE"]
    LONGITUDE = request.form["LONGITUDE"]
    community = request.form["community"]
    floor = request.form["floor"]
 
    #age = request.form["age"]
    #year = request.form["year"]
    #num_Axi = request.form["num_Axi"]
    X = np.array([[float(age), float(year), float(num_Axi)]])
    pred = model.predict_proba(X)[0][1]
    return \
    """
    <!DOCTYPE html>
    <html>
    <body>
    
    The Death Probability in Five Years is <br><br>
    {0}<br><br>
    
    <form action="/">
      <input type="submit" value="Recalculate">
    </form> 
    </body>
    </html>
    """.format(pred)


@app.route('/scoring', methods=['POST'])
def get_keywords():
    PROPERTY_TYPE = request.json["PROPERTY_TYPE"]
    CITY = request.json["CITY"]
    BEDS = request.json["BEDS"]
    BATHS = request.json["BATHS"]
    #LOCATION = request.json["LOCATION"]
    SQUARE_FEET = request.json["SQUARE_FEET"]
    LOT_SIZE = request.json["LOT_SIZE"]
    YEAR_BUILT = request.json["YEAR_BUILT"]
    DAYS_ON_MARKET = request.json["DAYS_ON_MARKET"]
    HOAperMONTH = request.json["HOAperMONTH"]
    LATITUDE = request.json["LATITUDE"]
    LONGITUDE = request.json["LONGITUDE"]
    community = request.json["community"]
    floor = request.json["floor"]
#    X = np.array([[float(age), float(year), float(num_Axi)]])
    X = np.array([[string(PROPERTY_TYPE), string(CITY), int(BEDS), int(BATHS), string(community), 
                   float(SQUARE_FEET),float(LOT_SIZE), int(YEAR_BUILT), int(DAYS_ON_MARKET), float(HOAperMONTH),
                   float(LATITUDE), float(LONGITUDE), string(community), string(floor)]])
    results = {"Estimated_House_Price":model.predict_proba(X)[0]}
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
