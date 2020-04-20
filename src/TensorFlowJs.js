import * as tfjs from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import carsData from "./data/carsData.json";

function TensorFlowJs() {
  console.log("Tensorflow is loaded > ", tfjs);

  // required by react component if no elements are created
  return null;
}

export default TensorFlowJs;
