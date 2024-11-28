import React, { useState, useEffect } from "react";
import Papa from "papaparse";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import "./App.css";

const handleFileUpload = (e, setData) => {
  const file = e.target.files[0];
  if (file) {
    Papa.parse(file, {
      header: true,
      complete: (result) => {
        const data = result.data.map((row) => ({
          sales_date: row.created,
          product_description: row.short_desc,
          quantity_sold: parseFloat(row.total_sold), 
        }));

        console.log("Parsed Data:", data);

        const validData = data.filter((row) => !isNaN(row.quantity_sold));
        console.log("Valid Data:", validData); 

        setData(validData); 
      },
    });
  }
};

function App() {
  const [salesData, setSalesData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [productFilter, setProductFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingData, setLoadingData] = useState(false); 

  const preprocessData = (data) => {
    const productMap = {};
    const dateMap = {};
    let productCounter = 0;
    let dateCounter = 0;

    const processedData = data.map((item) => {
      const dateIdx =
        dateMap[item.sales_date] ?? (dateMap[item.sales_date] = dateCounter++);
      const productIdx =
        productMap[item.product_description] ?? (productMap[item.product_description] = productCounter++);
      return {
        x: [dateIdx, productIdx],
        y: item.quantity_sold,
      };
    });

    
    const quantitySold = processedData.map((d) => d.y);
    const minQuantity = Math.min(...quantitySold);
    const maxQuantity = Math.max(...quantitySold);

    const normalizedData = processedData.map((d) => ({
      ...d,
      y: (d.y - minQuantity) / (maxQuantity - minQuantity),
    }));

    return { normalizedData, dateMap, productMap, minQuantity, maxQuantity };
  };

  const buildModel = async (data) => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, inputShape: [2], activation: "relu" }));
    model.add(tf.layers.dense({ units: 64, activation: "relu" })); 
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ optimizer: "adam", loss: "meanSquaredError" });

    const xs = tf.tensor(data.map((item) => item.x));
    const ys = tf.tensor(data.map((item) => item.y));

    await model.fit(xs, ys, { epochs: 300 }); 

    return model;
  };

  const forecastSales = async (model, productMap, dateMap, minQuantity, maxQuantity) => {
    setLoading(true);

    const futureStartDate = Math.max(...Object.values(dateMap)) + 1;
    const futureData = [];
    const productKeys = Object.keys(productMap);

    for (let i = 0; i < 6; i++) {
      productKeys.forEach((product) => {
        const productIdx = productMap[product];
        futureData.push([futureStartDate + i, productIdx]);
      });
    }

    const forecastXs = tf.tensor(futureData);
    const predictionsTensor = model.predict(forecastXs);
    const forecastValues = predictionsTensor.arraySync();

    const forecast = futureData.map((data, index) => {
      
      const predictedValue = forecastValues[index][0] * (maxQuantity - minQuantity) + minQuantity;

      return {
        month: `2024-${String(data[0]).padStart(2, "0")}`,
        product: Object.keys(productMap).find(
          (key) => productMap[key] === data[1]
        ),
        
        prediction: Math.max(predictedValue, 0),
      };
    });

    setPredictions(forecast);
    setLoading(false);
  };

  const renderLineChart = (data) => {
    const chartData = {
      values: data.map((d) => ({ x: d.month, y: d.prediction })),
      series: ["Predicted Sales"],
    };
  
    // Find the minimum and maximum values in the predictions
    const minPrediction = Math.min(...data.map((d) => d.prediction));
    const maxPrediction = Math.max(...data.map((d) => d.prediction));
  
    const container = document.getElementById("chart-container");
  
    // Render the chart with updated Y-axis scaling
    tfvis.render.linechart(container, chartData, {
      xLabel: "Month",
      yLabel: "Quantity Sold",
      height: 400,
      width: 700,
      // Set Y-axis domain to the range of min and max predictions
      yAxisDomain: [minPrediction, maxPrediction],
    });
  };
  

  const handleUploadAndForecast = async () => {
    const { normalizedData, dateMap, productMap, minQuantity, maxQuantity } =
      preprocessData(salesData);
    const model = await buildModel(normalizedData);
    await forecastSales(model, productMap, dateMap, minQuantity, maxQuantity);
    setProductFilter(Object.keys(productMap)[0]);
  };

  
  const filteredPredictions = predictions.filter(
    (p) => p.product === productFilter
  );

  useEffect(() => {
    if (predictions.length > 0 && !productFilter) {
      setProductFilter(predictions[0].product); 
      
    }
  }, [predictions, productFilter]);

  return (
    <div className="App">
      <h1>Sales Forecasting</h1>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => handleFileUpload(e, setSalesData)}
      />

      <button
        onClick={handleUploadAndForecast}
        disabled={loadingData || salesData.length === 0}
      >
        Forecast Sales
      </button>

      {loading && <p>Loading...</p>}

      <div>
        <label>Filter by Product:</label>
        <select
          value={productFilter || ""}
          onChange={(e) => setProductFilter(e.target.value)}
        >
          {predictions
            .map((p) => p.product)
            .filter((v, i, self) => self.indexOf(v) === i)
            .map((product) => (
              <option key={product} value={product}>
                {product}
              </option>
            ))}
        </select>
      </div>

      <div id="chart-container" style={{ marginTop: "20px" }}>
        {filteredPredictions.length > 0 && renderLineChart(filteredPredictions)}
      </div>
    </div>
  );
}

export default App;
