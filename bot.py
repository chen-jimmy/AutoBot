import os
import discord
from discord.ext import commands
from discord.ext.commands import has_permissions, MissingPermissions
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

#client = discord.Client()
bot = commands.Bot(command_prefix='$')

infractions = {}
strictness = 0

def is_bad(message):
	return message.find("chris") != -1

@bot.event
async def on_message(message):
	if message.author.bot:
		return

	if message.content == 'ping':
		await message.channel.send('pong')
		return

	if is_bad(message.content):
		if message.author.name in infractions:
			infractions[message.author.name] = infractions[message.author.name] + 1
		else:
			infractions[message.author.name] = 1

		await message.channel.send('This is your {}th warning {}'.format(infractions[message.author.name], message.author.name))
		return

	await bot.process_commands(message)

@bot.command()
async def test(ctx, arg):
	await ctx.send(arg)

@bot.command(pass_context=True)
@has_permissions(administrator=True)
async def set_strictness(ctx, arg):
	try:
		value = float(arg)
		strictness = value	
		await ctx.send("Set strictness to " + str(strictness))
	except:
		await ctx.send("Not a valid float!")

@set_strictness.error
async def set_strictness_error(ctx, error):
	await ctx.send("oops")

bot.run(TOKEN)
