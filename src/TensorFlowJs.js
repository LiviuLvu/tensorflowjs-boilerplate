import * as tfjs from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

function TensorFlowJs() {
  console.log("Tensorflow is loaded > ", tfjs);
  console.log("Tensorflow visor is loaded > ", tfvis);

  // required by react component if no elements are created
  return null;
}

export default TensorFlowJs;
