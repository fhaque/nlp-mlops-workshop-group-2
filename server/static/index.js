// import axios from 'axios';

const postForSummary = data => {
    return axios({
        method: 'post',
        url: '/predict',
        data: {
            data
        },
    })
    .then(res => res.data);
};

const onSubmit = e => {
    // clear summary
    // const summaryElement = document.getElementById('text-summary');
    // summaryElement.innerHTML = ''
    const embeddedTweetElement = document.getElementById('embedded-tweet');
    const summaryErrorElement = document.getElementById('summary-error');
    embeddedTweetElement.innerHTML = '';
    summaryErrorElement.innerHTML = '';

    // grab text content
    const inputElement = document.getElementById('text-content');
    const inputText = inputElement.value;

    // get prediction
    postForSummary(inputText).then(data => {
        // const summaryElement = document.getElementById('text-summary');
        // const tweetLinkElement = document.getElementById('tweet-url');
        const summaryErrorElement = document.getElementById('summary-error');
        const embeddedTweetElement = document.getElementById('embedded-tweet');

        if (data.error) {
            summaryErrorElement.innerHTML = `Error: (${data.error})`;
        } else {
            const { tweet, tweet_url, tweet_id } = data;
            // tweetLinkElement.href = tweet_url
            // summaryElement.innerHTML = tweet;
            // embeddedTweetElement.innerHTML = tweet_html;

            twttr.widgets.createTweet(
                tweet_id,
                embeddedTweetElement,
                {
                  theme: 'light'
                }
            );
            
        }
    })
    .catch(err => summaryElement.innerHTML = `Error: (${err})`);

    e.preventDefault();
    return false;
};

const formElement = document.getElementById('text-form');
formElement.addEventListener('submit', onSubmit);
