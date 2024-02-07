""" This is a very simple script to put the FAQ bot skeleton code on line
as a discord bot. If you follow the structure of the skeleton code, you should
not have to make any major changes here to get your bot on line.

However you should at least rename faq_bot_skeleton.py, which means you'll have
to change the import line below.

Note that the bot defined here will respond to EVERY message in every server it
is invited to. It is possible to have it only respond to messages that are
@ it, or only to private messages instead. I will leave it to you to figure that
out!

If you adapt this code, add yourself below as author and rewrite this header
comment. See the Resources folder on Canvas for documentation standards.

Modified by: Nicholas Milanovic, March 20, 2023
Sam Scott, Mohawk College, May 2021
"""
import discord
from chat_bot_shell_a3 import *

class MyClient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

    async def on_ready(self):
        print('Logged on as', self.user)

    async def on_message(self, message):
        # don't respond to ourselves
        if message.author == self.user:
            return

        # get the utterance and generate the response
        utterance = message.content
        intent = understand(utterance)
        response = generate(intent, utterance)
        if message.content == "hello".casefold():
            await message.channel.send("Hello! I know stuff about LeBron James. When you're done talking, just say 'goodbye'.")
        if message.content == "goodbye".casefold():
            await message.channel.send("Nice talking to you!")
            await message.channel.purge()   # purge function used to clear the chat
        # send the response
        await message.channel.send(response)

client = MyClient()
with open("bot_token.txt") as file:
    token = file.read()

client.run(token)