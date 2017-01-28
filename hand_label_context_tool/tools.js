var data_labels = [[0,0,1]];
var sentence_indices = [];
var sentences = [];
var key_word_index_pairs = [];
var current_index = 0;
var dict = [];

var pair_dict = {};

if(typeof(String.prototype.trim) === "undefined")
{
    String.prototype.trim = function() 
    {
        return String(this).replace(/^\s+|\s+$/g, '');
    };
}

function arraysEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (a.length != b.length) return false;

  // If you don't care about the order of the elements inside
  // the array, you should sort both arrays here.

  for (var i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function prevSentence() {
    if (current_index > 0) {
        current_index--; 
        update();
    } else {
        alert('This is the first sentence of the dataset.');
    }
}

function nextSentence() {
    if (current_index < sentences.length - 1) {
        current_index++; 
        if (data_labels.length <= current_index) {
            data_labels.push([0,0,1]);
        }
        update();
    } else {
        alert('You have reached the end of the dataset.');
    }
}

function onLoad(dict_file, unlabeled_data_file,labels_file){
    loadDictionary(dict_file);
    loadUnlabeledData(unlabeled_data_file);
    loadLabels(labels_file);
    update();
}

function loadDictionary(file_location) {
    var dict_text = readTextFile(file_location);
    dict = dict_text.split(/\r\n|\n/);
}

function loadUnlabeledData(file_location) {
    var text = readTextFile(file_location);
    var sentence_indices_text = text.split(/\r\n|\n/);
    for (var i=0; i<sentence_indices_text.length; i++) {
        var current_line = sentence_indices_text[i].split(' ');
        if (current_line.length <= 3){
            break;
        }
        var current_sentence_indices = current_line.slice(0, current_line.length-2);
        sentence_indices.push(current_sentence_indices);
        sentences.push(current_sentence_indices.map((word)=>{
            if (word == '0') {
                return '';
            } else {
                return dict[word];
            }
        }));
        key_word_index_pairs.push([current_line[current_line.length - 2],current_line[current_line.length - 1]]);
    }

}

function loadLabels(file_location) {
    var text = readTextFile(file_location);
    if (text != null){
        data_labels = [];
        var sentence_indices_text = text.split(/\r\n|\n/);
        for (var i=0; i<sentence_indices_text.length; i++) {
            var current_line = sentence_indices_text[i].split(' ');
            if (current_line.length != 3){
                break;
            }
            data_labels.push([parseInt(current_line[0]),parseInt(current_line[1]),parseInt(current_line[2])]);
            var temp_index = data_labels.length - 1;
            var first_key_phrase = sentences[temp_index][key_word_index_pairs[temp_index][0]];
            var second_key_phrase = sentences[temp_index][key_word_index_pairs[temp_index][1]];
            pair_dict[[first_key_phrase, second_key_phrase]] = data_labels[temp_index];
        }
        current_index = data_labels.length - 1;
    }
}

function readTextFile(file)
{
    xmlhttp = new XMLHttpRequest();
    xmlhttp.open("GET",file,false);
    xmlhttp.send(null);
    if (xmlhttp.status==404){
        return null;
    }else {
        var fileContent = xmlhttp.responseText;
        return fileContent;
    }
}

function save() {
    save_txt = '';

    for (var i = 0; i < data_labels.length; i++) {
        save_txt += data_labels[i].join(' ');
        if (i != data_labels.length - 1) {
            save_txt += '\n';
        }
    }

    var blob = new Blob([save_txt], {type: "text/plain;charset=utf-8"});
    saveAs(blob, "test_cs_labels_combined.txt");
}

function update(){
    var text = document.getElementById("text");
    var updated_text = '';
    for (var i = 0; i < sentences[current_index].length; i++) {
        if (i == key_word_index_pairs[current_index][0] || i == key_word_index_pairs[current_index][1]) {
            updated_text+=" <span class='highlight'>";
            updated_text+=sentences[current_index][i];
            updated_text+="</span>";
        }
    }
    updated_text+='<br>';
    for (var i = 0; i < sentences[current_index].length; i++) {
        if (i == key_word_index_pairs[current_index][0] || i == key_word_index_pairs[current_index][1]) {
            updated_text+="<span class='highlight'>";
        }
        updated_text+=sentences[current_index][i];
        if (i == key_word_index_pairs[current_index][0] || i == key_word_index_pairs[current_index][1]) {
            updated_text+="</span>";
        }
        if (i != sentences[current_index].length - 1){
            updated_text+=' ';
        }
    }

    // Now update the radio button
    var first_key_phrase = sentences[current_index][key_word_index_pairs[current_index][0]];
    var second_key_phrase = sentences[current_index][key_word_index_pairs[current_index][1]];

    document.getElementById("radio_option_0").innerHTML = "'" + first_key_phrase + "' is-a '" + second_key_phrase + "'";
    document.getElementById("radio_option_1").innerHTML = "'" + second_key_phrase + "' is-a '" + first_key_phrase + "'";

    updated_text+="<br>";
    if ((current_index == data_labels.length - 1) && ([first_key_phrase, second_key_phrase] in pair_dict)) {
        if (data_labels[current_index] != pair_dict[[first_key_phrase, second_key_phrase]]) {
        // Potential bug: if to last page, change answer, then back to previous one, then to last page again, the answer will not be saved and you have to chagne it again.
            data_labels[current_index] = pair_dict[[first_key_phrase, second_key_phrase]];
        } 
        updated_text+="Seen before!";
        // else {
        //     updated_text+="Seen before but answer different! Previous answer was: " + pair_dict[[first_key_phrase, second_key_phrase]];
        // }
    }

    text.innerHTML = updated_text;

    if (data_labels[current_index][0] == 1) {
        document.getElementById("radio_option_0").checked = true;
    } else if (data_labels[current_index][1] == 1) {
        document.getElementById("radio_option_1").checked = true;
    } else if (data_labels[current_index][2] == 1) {
        document.getElementById("radio_option_2").checked = true;
    } else {
        alert('There is a bug during update in the radio selection.');
    }

}

function radioSelected(option){
    switch(option) {
    case "0":
        data_labels[current_index] = [1,0,0];
        break;
    case "1":
        data_labels[current_index] = [0,1,0];
        break;
    case "2":
        data_labels[current_index] = [0,0,1];
        break;
    default:
        alert('There is a bug in the radio selection');
    }
}