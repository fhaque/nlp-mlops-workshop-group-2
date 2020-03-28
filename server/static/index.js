// import axios from 'axios';



const postForSummary = data => {
    summaryElement = document.getElementById('text-summary');
    summaryElement.innerHTML = ''
    return axios({
        method: 'post',
        url: '/predict',
        data: {
            data
        },
    })
    .then(res => parsedResponse = res.data);
};

const onSubmit = e => {
    console.log("Submitting...");
    e.preventDefault();
    inputElement = document.getElementById('text-content');
    inputText = inputElement.innerHTML;
    console.log(inputText);
    postForSummary(inputText).then(data => {
        summaryElement = document.getElementById('text-summary');
        summaryElement.innerHTML = parsedResponse;
    });

    return false;
};


const formElement = document.getElementById('text-form');
formElement.addEventListener('submit', onSubmit);