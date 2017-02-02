$(function() {

        $("#submit").prop("disabled",false);

        $("#submit").click( function(){
            submit();
        });


//--- functions

        startPaint = function(){
            $("#submit").prop("disabled",true);
            console.log("start");
        }
        endPaint = function(){
            $("#submit").prop("disabled",false);
            console.log("finish");
        }

        submit = function(){
            var send_data = true;
            startPaint();
            var ajaxData = new FormData();
            var username = $("#username").val();
            if (username == ''){
                alert("Please enter a username.");
                send_data = false;
            }
            ajaxData.append('username', $("#username").val() );
            if (document.getElementById("instance_id").innerHTML != '') {
                var label = getRadioSelected();
                if (label == -1) {
                    alert("Please select an option below and then hit submit.");
                    send_data = false;
                } else {
                    ajaxData.append('instance_id', document.getElementById("instance_id").innerHTML );
                    ajaxData.append('comments', document.getElementById("comments").value );
                    ajaxData.append('label', label);
                }
            }
            if (send_data) {
                $.ajax({
                    url: "/post",
                    data: ajaxData,
                    cache: false,
                    contentType: false,
                    processData: false,
                    type: 'POST',
                    dataType:'json',
                    complete: function(data) {
                            console.log("uploaded")
                            var now = new Date().getTime();
                            var message = JSON.parse(data.responseText);
                            if (message.success == false) {
                                alert(message.error_message);
                            } else {
                                document.getElementById("user_input_area").hidden = false;
                                document.getElementById("key_phrase_pair").innerHTML = message.key_phrase_pair;
                                document.getElementById("sentence").innerHTML = message.sentence;
                                document.getElementById("instance_id").innerHTML = message.instance_id;
                                document.getElementById("comments").value = '';
                                if (message.user_num_labeled == 3) {
                                    alert("You've labeled 10 instances! Thank you for contributing to my research. :) As a reward, here are some links below to an NLP course which I found helpful.");
                                    document.getElementById("rewards").hidden = false;
                                    document.getElementById("rewards").innerHTML = 'Interesting NLP Deep learning course syllabus: http://cs224d.stanford.edu/syllabus.html. <br> videos: https://www.youtube.com/playlist?list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG';
                                } else if (message.user_num_labeled == 4) {
                                    alert("You've labeled 50 instances! As a reward, check the page for links to one of my all time favourite deep learning courses.");
                                    document.getElementById("rewards").hidden = false;
                                    document.getElementById("rewards").innerHTML = 'Interesting NLP Deep learning course syllabus: http://cs224d.stanford.edu/syllabus.html. <br> videos: https://www.youtube.com/playlist?list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG <br> All time favourite CNN course: http://cs231n.stanford.edu/syllabus.html <br> videos: https://www.youtube.com/playlist?list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG';
                                } else if (message.user_num_labeled == 5) {
                                    alert("You've labeled 100 instances! Here's my final reward: A small website I found lately that uses deep learning to color Anime sketches. Thanks for supporting my research.");
                                    document.getElementById("rewards").hidden = false;
                                    document.getElementById("rewards").innerHTML = 'Interesting NLP Deep learning course syllabus: http://cs224d.stanford.edu/syllabus.html. <br> videos: https://www.youtube.com/playlist?list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG <br> All time favourite CNN course: http://cs231n.stanford.edu/syllabus.html <br> videos: https://www.youtube.com/playlist?list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG <br> An interesting website that color Anime sketches: http://paintschainer.preferred.tech/ <br> Try it with this sketch for example: http://www.pixiv.net/member_illust.php?mode=medium&illust_id=31274285 <br> English translation of an overview of the project: http://qiita.com/jerryli27/items/f526a7d5b69ae758a3a6';
                                }
                            }
                    }
                  });
            }
            endPaint();
        }

        function getRadioSelected(){
            if (document.getElementById("radio_option_0").checked) {
                return 0;
            }
            else if (document.getElementById("radio_option_1").checked) {
                return 1;
            }
            else if (document.getElementById("radio_option_2").checked) {
                return 2;
            }
            else if (document.getElementById("radio_option_3").checked) {
                return 3;
            }
            return -1;
        }


});
