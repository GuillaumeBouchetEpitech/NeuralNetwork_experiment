
'use strict'

//
//
// TrainingData

function TrainingData (str_content) {
	this._arr_lines = str_content.trim().split('\n');
};

TrainingData.prototype.isEof = function() {
	return (this._arr_lines.length == 0)
}

TrainingData.prototype._extract_from_data = function (str_pattern, arr_output) {

	arr_output.length = 0; // clear array

	var str_value = this._arr_lines.shift().trim();
	var arr_elements = str_value.split(' ');

	if (arr_elements.length > 0 &&
		arr_elements[0] == str_pattern)
	{
		for (var i = 1; i < arr_elements.length; ++i)
			arr_output.push( parseFloat(arr_elements[i]) );
	}

	return arr_output.length;
}

TrainingData.prototype.getTopology = function(arr_topology) {
	return this._extract_from_data('topology:', arr_topology);
}

TrainingData.prototype.getNextInputs = function(arr_inputVals) {
	return this._extract_from_data('in:', arr_inputVals);
}

TrainingData.prototype.getTargetOutputs = function(arr_targetOutputVals) {
	return this._extract_from_data('out:', arr_targetOutputVals);
}

// TrainingData
//
//

module.exports = TrainingData;
