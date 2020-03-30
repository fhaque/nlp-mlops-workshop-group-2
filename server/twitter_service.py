import json

import twitter
import requests

class TwitterService():

    def __init__(self):
        with open('secrets/twitter_keys.json', 'r') as f:
            keys = json.load(f)

        self._api = self._connect_to_twitter(
            keys['consumer_key'],
            keys['consumer_secret'],
            keys['access_token'],
            keys['access_token_secret']
        )

    def _connect_to_twitter(self, consumer_key, consumer_secret, access_token, access_token_secret, **kwargs):
        api = twitter.Api(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token_key=access_token,
            access_token_secret=access_token_secret,
            **kwargs
        )
        return api

    def tweet(self, message):
        status = self._api.PostUpdate(message)

        return status

    def create_tweet_url(self, status):
        return "https://twitter.com/{}/status/{}".format(status.user.screen_name, status.id_str)

    def get_tweet_embed_html(self, status):
        return requests.get(
            'http://publish.twitter.com/oembed',
            params={
                "url": self.create_tweet_url(status),
            },
        ).json()['html']

    def get_tweet_id(self, status):
        return status.id_str