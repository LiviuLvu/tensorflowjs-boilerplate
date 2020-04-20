import React from "react";
import logo from "./img/logo.svg";
import tfjslogo from "./img/tensorflow-js-logo-social.png";
import "./App.css";
import TensorFlowJs from "./TensorFlowJs";

function App() {
  return (
    <div className="App">
      <TensorFlowJs />
      <header className="App-header">
        <img src={tfjslogo} className="tf-logo" alt="logo" />
        <p>
          <img src={logo} className="react-logo" alt="logo" />
          Experimenting with Tensorflow.js and React
        </p>
      </header>
    </div>
  );
}

export default App;
