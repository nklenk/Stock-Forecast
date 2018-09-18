#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 14:23:11 2018

@author: neilklenk
"""



# Import smtplib for the actual sending function
import smtplib


# Here are the email package modules we'll need
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def email_results():
    sender = 'neil.klenk@gmail.com'
    receivers = ['neil.klenk@gmail.com']#, 'johnk9000@gmail.com']
    
    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg['Subject'] = 'Todays Stock Preds'
    # me == the sender's email address
    # family = the list of all recipients' email addresses
    msg['From'] = 'neil.klenk@gmail.com'
    msg['To'] = 'neil.klenk@gmail.com' #COMMASPACE.join(family)
    msg.preamble = 'Our family reunion'
    
    
    filename = "stock_plots.html"
    f = open(filename)
    attachment = MIMEText(f.read(), _subtype='html')
    attachment.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(attachment)
    
    gmail_sender = 'neil.klenk@gmail.com'
    gmail_passwd = 'earMark23!!'
    
    # Send the email via our own SMTP server.
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(gmail_sender, gmail_passwd)
    print ("Successfully sent email")
    
    server.sendmail(sender, receivers, msg.as_string()) #
    server.quit()

