window.onload = function(){
    const http = 'http://127.0.0.1:5000'
    //about input
    const input_text = document.getElementById("input_text")
    const model = document.getElementById("model")
    const btn_analyze = document.getElementById("btn-analyze")

    //about ouput
    const ouput = document.getElementById("output")
    const box_output = document.getElementById("box-output")

    // init random input_text 
    fetch(http+"/initinput")
      .then(response => response.json())
      .then(data => {
        input_text.value = data.init_input
      });
    
    // send data and get predict
    btn_analyze.addEventListener("click", ()=>{
        // predict
        box_output.style.display = "none"
        fetch(http+"/predict",{
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({"input_text":input_text.value, "model":model.value}),
        })
        .then(response => response.json())
        .then(data => {
            ouput.innerHTML = data.predict
            box_output.style.display = "block"
        });
    })

    // btn_analyze.addEventListener("click", ()=>{
    //     // test prediction
    //     box_output.style.display = "none"
    //     fetch(http+"/testpredict")
    //     .then(response => response.json())
    //     .then(data => {
    //         ouput.innerHTML = data.predict
    //         box_output.style.display = "block"
    //     });
    // })
}