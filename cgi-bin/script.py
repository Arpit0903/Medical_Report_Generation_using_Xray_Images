import cgi
form = cgi.FieldStorage()
x = form.getvalue('myfile')
print(x)