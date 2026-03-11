async function uploadFile(){
    let fileInput = document.getElementById("file")
    let file = fileInput.files[0]
    let button = document.getElementById("uploadBtn")
    let status = document.getElementById("uploadStatus")

    if(!file){
    alert("Please select a file")
    return
    }

    // disable button
    button.disabled = true
    button.innerText = "Processing..."
    status.innerText = "Document is being processed..."

    let formData = new FormData()
    formData.append("file", file)

    try{
        let response = await fetch("/upload",{method:"POST",body:formData})
        let data = await response.json()
        document.getElementById("uploadResult").innerText =
        JSON.stringify(data,null,2)
        status.innerText = "Processing complete ✅"
    }
    catch(error){
        status.innerText = "Error processing document ❌"
        console.error(error)
    }

    // enable button again
    button.disabled = false
    button.innerText = "Upload"
}

async function searchDocs(){
    let query=document.getElementById("searchBox").value
    let patient=document.getElementById("patientSelect").value
    let response=await fetch(`/search?q=${query}&patient_id=${patient}`)
    let data=await response.json()
    document.getElementById("searchResult").innerText=
    JSON.stringify(data,null,2)

}


async function askQuestion(){
    let question=document.getElementById("questionBox").value
    let patient=document.getElementById("patientSelect").value
    let response=await fetch("/ask",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(
            {
            question:question,
            patient_id:patient
            }
        )
    })
    let data=await response.json()
    document.getElementById("answerResult").innerText=data.answer
}

