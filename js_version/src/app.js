

'use strict'

// var fs = require('fs');

var TrainingData = require("./TrainingData.js");
var NeuralNetwork = require("./NeuralNetwork.js");


//
//
// MAIN

var curr_timeout = null;

function runNeuralNet() {

    if (curr_timeout)
        clearTimeout(curr_timeout);

    var elem_logs = document.getElementById("logs_area");

    elem_logs.value = ""; // clear logs

    function print(msg) {
        if (msg)
            elem_logs.value += msg;
        elem_logs.value += '\n';

        elem_logs.scrollTop = elem_logs.scrollHeight
    }

    function showVectorVals(prefix, arr_values)
    {
        var str_msg = prefix + " ";
        for (var i = 0; i < arr_values.length; ++i)
            str_msg += arr_values[i] + " ";

        print(str_msg);
    }

    //
    //
    // canvas drawing

    var main_canvas = document.getElementById("main-canvas");
    var ctx = main_canvas.getContext("2d");

    function clear () {
        ctx.clearRect(0, 0, main_canvas.width, main_canvas.height);
    }

    function drawLine (x1, y1, x2, y2, color, size)
    {
        ctx.strokeStyle = color;
        ctx.lineWidth = size || 1;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    };

    function draw_circle (x, y, radius, color) {
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI, false);
        ctx.fillStyle = color;
        ctx.fill();
    };

    // canvas drawing
    //
    //

    { // execution

        // var str_data = fs.readFileSync('../src/trainsample/out_xor.txt', 'utf8');
        var str_data = document.getElementById("output_area").value;

        var trainData = new TrainingData(str_data);


        var arr_topology = [];
        trainData.getTopology(arr_topology);
        print('arr_topology=' + JSON.stringify(arr_topology));

        var myNeuralNetwork = new NeuralNetwork(arr_topology);

        function updateCanvas(in_NeuralNetwork) {

            // clear

            clear();

            // borders

            drawLine(0,0, main_canvas.width, 0, "#0000ff", 10);
            drawLine(main_canvas.width,0, main_canvas.width, main_canvas.height, "#0000ff", 10);
            drawLine(main_canvas.width, main_canvas.height, 0, main_canvas.height, "#0000ff", 10);
            drawLine(0, main_canvas.height, 0, 0, "#0000ff", 10);

            // topology - connection(s)

            var arr_weights = in_NeuralNetwork.getWeights();
            var index_weights = 0;
            var max_weight = 0;
            for (var i = 0; i < arr_weights.length; ++i)
                max_weight = Math.max(arr_weights[i], max_weight);

            var step_x = (main_canvas.width / (arr_topology.length + 1))

            for (var x = 0; x < arr_topology.length - 1; ++x)
            {
                var step_y1 = (main_canvas.height / (arr_topology[x] + 1))
                var step_y2 = (main_canvas.height / (arr_topology[x+1] + 1))

                for (var y1 = 0; y1 < arr_topology[x]; ++y1)
                    for (var y2 = 0; y2 < arr_topology[x+1]; ++y2)
                    {
                        var val_weight = arr_weights[index_weights++] / max_weight * 10;
                        var color = (val_weight < 0 ? "#0000ff" : "#ff0000");

                        if (val_weight < 0)
                            val_weight = -val_weight;

                        if (val_weight < 1)
                            val_weight = 1;

                        drawLine(step_x + step_x * x,step_y1 + step_y1 * y1, step_x + step_x * (x+1),step_y2+step_y2*y2, color, val_weight);
                    }
            }

            // topology - neuron(s)

            for (var x = 0; x < arr_topology.length; ++x)
            {
                var step_y = (main_canvas.height / (arr_topology[x] + 1))

                for (var y = 0; y < arr_topology[x]; ++y)
                {
                    draw_circle(step_x + step_x * x, step_y + step_y * y, 13, "#00ff00");
                }
            }

        } // updateCanvas()


        var arr_inputVals = [];
        var arr_resultVals = [];
        var arr_targetVals = [];
        var trainingPass = 0;

        curr_timeout = setTimeout(tick, 0);

        function tick() {

            if (trainData.isEof())
            {
                curr_timeout = setTimeout(done, 0);
                return;
            }

            // while (!trainData.isEof() && trainingPass < 5)
            // while (!trainData.isEof())
            do
            {
                // trainData.getNextInputs(arr_input);
                // print('arr_input=', arr_input);

                // trainData.getTargetOutputs(arr_output);
                // print('arr_output=', arr_output);


                ++trainingPass;
                print();
                print("Pass " + trainingPass);

                // Get new input data and feed it forward:
                if (trainData.getNextInputs(arr_inputVals) != arr_topology[0])
                    break;

                showVectorVals("Inputs:", arr_inputVals);
                myNeuralNetwork.feedForward(arr_inputVals);

                // Collect the net's actual output results:
                myNeuralNetwork.getResults(arr_resultVals);
                showVectorVals("Outputs:", arr_resultVals);

                // Train the net what the outputs should have been:
                trainData.getTargetOutputs(arr_targetVals);
                showVectorVals("Targets:", arr_targetVals);
                if (arr_targetVals.length != arr_topology[arr_topology.length - 1])
                    throw new Error('(arr_targetVals.length != arr_topology.back())');


                myNeuralNetwork.backProp(arr_targetVals);


                updateCanvas(myNeuralNetwork);


                // Report how well the training is working, average over recent samples:
                print("Net current error: " + myNeuralNetwork.getError());
                print("Net recent average error: " + myNeuralNetwork.getRecentAverageError());

                if (trainingPass > 100 && myNeuralNetwork.getRecentAverageError() < 0.05)
                {
                    print("average error acceptable -> break");

                    curr_timeout = setTimeout(done, 10);

                    break;
                }

                curr_timeout = setTimeout(tick, 0);
            }
            while(false)
        }

        function done() {

            print();
            print("Done");
            print();

            if (arr_topology.length < 2 ||
                arr_topology[0] != 2)
            {
                print("Unexpected topology, no test");
            }
            else
            {
                print("TEST");
                print();

                var dblarr_test = [ [0,0], [0,1], [1,0], [1,1] ];

                for (var i = 0; i < 4; ++i)
                {
                    arr_inputVals.length = 0;
                    arr_inputVals.push(dblarr_test[i][0]);
                    arr_inputVals.push(dblarr_test[i][1]);

                    myNeuralNetwork.feedForward(arr_inputVals);
                    myNeuralNetwork.getResults(arr_resultVals);

                    showVectorVals("Inputs:", arr_inputVals);
                    showVectorVals("Outputs:", arr_resultVals);

                    print();
                }

                print("/TEST");
            }
        }

    } // execution

}


(function(){

    document.getElementById("runNeuralNet").onclick = runNeuralNet;

})();

// MAIN
//
//

