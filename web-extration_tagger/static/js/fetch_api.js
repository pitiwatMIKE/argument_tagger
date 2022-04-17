window.onload = function(){
    const http = 'http://127.0.0.1:5000'
    //about input
    const input_text = document.getElementById("input_text")
    const model = document.getElementById("model")
    const btn_analyze = document.getElementById("btn-analyze")
    const btnPredictAll = document.getElementById("btn-predictall")

    //about ouput
    const ouput = document.getElementById("output")
    const box_output = document.getElementById("box-output")

    // init random input_text 
    fetch(http+"/initinput")
      .then(response => response.json())
      .then(data => {
        input_text.value = data.init_input
        input_text2.value = data.init_input
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

    // predict all model
    const outputCRF = document.getElementById('output-crf')
    const outputLSTM = document.getElementById('output-lstm')
    const outputBiLSTM = document.getElementById('output-bilstm')
    const outputBiLSTM_CRF = document.getElementById('output-bilstmcrf')
    const wangchanberta = document.getElementById('output-wangchanberta')
    const show = document.getElementById('predict-output-all')
    
    btnPredictAll.addEventListener("click", ()=>{
      fetch(http+"/predict_all",{
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({"input_text":input_text2.value}),
    })
    .then(response => response.json())
    .then(data => {
        show.style.display = "block"
        outputCRF.innerHTML = data.crf
        outputLSTM.innerHTML = data.lstm
        outputBiLSTM.innerHTML = data.bilstm
        outputBiLSTM_CRF.innerHTML = data.bilstm_crf
        wangchanberta.innerHTML = data.wangchanberta
        console.log(data);
      });
    })
}
