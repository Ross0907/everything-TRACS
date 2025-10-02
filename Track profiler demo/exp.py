# print(abs(-1))
# print(type(print(True)))

# # Sample inputs (# note: The values given in the prefix code(grey) will be changed by the autograder according to the testcase while running them.
# a = 4
# b = 7
# price, discount_percent = 80, 2
# total_mins = 270
# # <eoi>

# output1 = int(a + b) # int: sum of a and b
# output2 = 2*output1 # int: twice the sum of a and b
# output3 = abs(a-b) # int: absolute difference between a and b
# output4 = abs(output1 - (a*b)) # int: absolute difference between sum and product of a and b

# # Find discounted price given price and discount_percent
# # input variables : price: int, discount_percent: float
# discounted_price = float(price - (discount_percent/100)*price) # float

# # Round the discounted_price
# rounded_discounted_price = int(discounted_price) # int

# # Find hrs and mins given the total_mins
# # input variables : total_mins
# hrs = total_mins//60 # int: hint: think about floor division operator
# mins = total_mins - (hrs*60)# int

# print(output1)
# print(output2)
# print(output3)
# print(output4)
# print(discounted_price)
# print(rounded_discounted_price)
# print(hrs)
# print(mins)




# print(len(str(49)))


# # Sample inputs (# note: The values given in the prefix code(grey) will be changed by the autograder according to the testcase while running them.
# a = 1234

# price1, discount1 = 150, 25 # for offer1
# price2, discount2 = 200, 45 # for offer2

# # Assume discount is given in percentages

# # <eoi>

# output1 = a>=5 # bool: True if a greater than or equal to 5

# output2 = (a%5==0) # bool: True if a is divisible by 5

# output3 = (a%2 != 0) and a<10 # bool: True if a is odd number less than 10

# output4 = (a%2 != 0) and a<10 and a>-10 # bool: True if a is an odd number within the range -10 and 10

# output5 = len(str(a))%2==0 and  len(str(a))<=10# bool: True if a has even number of digits but not more than 10 digits

# is_offer1_cheaper = (float(price1 - (discount1/100)*price1))<(float(price2 - (discount2/100)*price2)) # bool: True if the offer1 is strictly cheaper

# print(output1)
# print(output2)
# print(output3)
# print(output4)
# print(output5)
# print(is_offer1_cheaper)

# s = "hello pyhton"
# course_code = "24t2cs1002" # 24 - year, t2 - term 2, cs1002 - course id
# # <eoi>

# output1 = s[2] # str: get the third character of s

# output2 = s[-4] # str: get the fourth last character of s

# output3 = s[:3] # str: get the first 3 characters of s

# output4 = s[::2] # str: get every second character of s

# output5 = s[-3:] # str: get the last 3 characters of s

# output6 = s[::-1] # str: get the reverse of s

# course_term = course_code[2:4] # int: get the term of the year as number from course_code
# course_year = course_code[:2] # int: get the year as two digit number from course_code

# print(output1)
# print(output2)
# print(output3)
# print(output4)
# print(output5)
# print(output6)
# print(course_term)
# print(course_year)


# # Sample inputs (# note: The values given in the prefix code(grey) will be changed by the autograder according to the testcase while running them.
# word1 = "Wingardium" # str
# word2 = "Leviyosa" # str
# word3 = "Silver" # str
# sentence = "Learning python is fun"
# n1 = 6 # int
# n2 = 4 # int
# # <eoi>

# output1 = word1 + " " + word2 # str: join word1 and word2 with space in between

# output2 = word1[:4] + "-" + word2[-4:] # str: join first four letters of word1 and last four letters of word 2 with a hyphen "-" in between

# output3 = word3 + " " + str(n1) # str: join the word3 and n1 with a space in between

# output4 = "-" * 50 # str: just the hypen "-" repeated 50 times

# output5 = "-" * n2 # str: just the hypen "-" repeated n2 times

# output6 = str(n1) * n2 # str: repeat the number n1, n2 times

# are_all_words_equal = (word1 == word2 == word3) # bool: True if all three words are equal

# is_word1_comes_before_other_two = (word1 < word2) and (word1 < word3) # bool: True if word1 comes before word2 and word3 assume all words are different

# has_h = "h" in word1 # bool: True if word1 has the letter h

# ends_with_a = word1.endswith("a") or word1.endswith("A") # bool: True if word1 ends with letter a or A

# has_the_word_python = "python" in sentence # bool: True if the sentence has the word python

# print(output1)
# print(output2)
# print(output3)
# print(output4)
# print(output5)
# print(output6)
# print(are_all_words_equal)
# print(is_word1_comes_before_other_two)
# print(has_h)
# print(ends_with_a)
# print(has_the_word_python)


# age = int(input()) # int: Read a number as integer from standard input
# dob = input() # str: Read a string of format dd/mm/yy from standard input
# day, month, year = int(dob[:2]),int(dob[3:5]),int(dob[6:]) # int, int, int: Get the correct parts from dob as int

# fifth_birthday = f"{day}/{month}/{year + 5}" # str: fifth birthday formatted as day/month/year

# last_birthday = f"{day}/{month}/{year - 1}" # str: last birthday formatted as day/month/year

# tenth_month = f"{day}/{month}/{year + 1}" # str: dob same day after 10 months formatted as day/month/year

# # print tenth_month, fifth_birthday and last_birthday in same line separated by comma and a space
# print(f"{tenth_month}, {fifth_birthday}, {last_birthday}")

# weight = float(input()) # float: Read a number as float from stdin(Standard input)

# weight_readable = f"{int(weight)} kg {int((weight % 1) * 1000)} grams" # str: reformat weight of format 55 kg 250 grams

# # print weight_readable
# print(weight_readable)


print(type("555"[2]))
print(3 + ((4 * 5) // 2))
word = '138412345678901938'
print(word[4:13] == '123456789')
x = 2 ** 5
x1 = x // 2
x2 = x1 // 2
x3 = x2 // 2
x4 = x3 // 2
x5 = x4 // 2
print(x1 + x2 + x3 + x4 + x5)