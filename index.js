const Discord = require('discord.js');
 const client = new Discord.Client();

client.on('ready', () => {
 console.log(`Logged in as ${client.user.tag}!`);
 });

client.on('message', msg => {
	 if (msg.content === 'ping') {
	 msg.reply('pong');
	 }
 });

var fs = require('fs')
var textByline = fs.readFileSync('.apikey').toString().split("\n")

client.login(textByline[0])
//client.login('token');
