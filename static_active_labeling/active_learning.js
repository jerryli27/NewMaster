$(function() {
        image_id = "test_id"

        $("#submit").prop("disabled",false);

        $("#submit").click( function(){
            submit();
        });


//--- functions 

        function uniqueid(){
            var idstr=String.fromCharCode(Math.floor((Math.random()*25)+65));
            do {                
                var ascicode=Math.floor((Math.random()*42)+48);
                if (ascicode<58 || ascicode>64){
                    idstr+=String.fromCharCode(ascicode);    
                }                
            } while (idstr.length<32);
            return (idstr);
        } 

        startPaint = function(){
            $("#submit").prop("disabled",true);
            console.log("coloring start");
        }
        endPaint = function(){
            $("#submit").prop("disabled",false);
            console.log("coloring finish");
        }

        submit = function(){
            startPaint()
            var ajaxData = new FormData();
            ajaxData.append('username', $("#username").val() );
            if (document.getElementById("instance_id").innerHTML != '') {
                ajaxData.append('instance_id', document.getElementById("instance_id").innerHTML );
                ajaxData.append('label', getRadioSelected() );
            }

            $.ajax({
                url: "/post",
                data: ajaxData,
                cache: false,
                contentType: false,
                processData: false,
                type: 'POST',
                dataType:'json',
                complete: function(data) {
                        //location.reload();
                        console.log("uploaded")
                        var now = new Date().getTime();
//                        $('#output').attr('src', '/static/images/out/'+image_id+'_0.jpg?' + now);
//                        $('#output_min').attr('src', '/static/images/out_min/'+image_id+'_0.png?' + now);
                        var message = JSON.parse(data.responseText);
                        if (message.success == false) {
                            alert(message.error_message);
                        } else {
                            document.getElementById("key_phrase_pair").innerHTML = message.key_phrase_pair;
                            document.getElementById("sentence").innerHTML = message.sentence;
                            document.getElementById("instance_id").innerHTML = message.instance_id;
                        }
                        endPaint()
                }
              });
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
            alert('There is a bug in the radio selection');
            return 3;
        }


});
