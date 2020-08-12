import xml.etree.ElementTree as ET
import os


def rename_to_xml(filename):
    #==== CHANGE FILENAME TO EXTENSION .XML ====#
    # print(os.listdir())
    xml = filename.replace('.rpx', '.xml')
    if filename in os.listdir():
        os.rename(filename, xml)
    # print(xml)
    return xml


def xml_edit(filename):
    # ==== EDIT TAGS TO ADD PREFIXES ==== #
    # Open original file
    rename_to_xml(filename)
    et = ET.parse(filename)
    root = et.getroot()
    parent = root.find("{AIR.InformationModel.Reinsurance}Treaties")
    # parent = et.find('Treaties')
    parent = parent.find('{AIR.InformationModel.Reinsurance}ReinsuranceTreaty')

    tags = parent.find('{AIR.InformationModel.Reinsurance}AppliesToLineOfBusiness')
    # print(parent.getchildren())
    # print(tags)
    # print(tags.text)
    text = tags.text
    og_text = text

    # tag = et.getroot().('AppliesToLineOfBusiness')
    # print(tag)
    # Append new tag: <a x='1' y='abc'>body text</a>
    # new_tag = xml.etree.ElementTree.SubElement(et.getroot(), 'AppliesToLineOfBusiness')

    lob_split = text.split(',')
    new_text = []
    for lob in lob_split:
        new_text.append('A_' + lob)
        new_text.append('L_' + lob)
        new_text.append('O_' + lob)

    tags.text = ','.join(new_text)
    tags.set('updated', 'yes')

    # Write back to file
    new_filename = filename.replace('.xml', '_updatedLOB.rpx')
    et.write(new_filename)

    # check

    et = ET.parse(new_filename)
    root = et.getroot()
    parent = root.find("{AIR.InformationModel.Reinsurance}Treaties")
    # parent = et.find('Treaties')
    parent = parent.find('{AIR.InformationModel.Reinsurance}ReinsuranceTreaty')

    new_tags = parent.find('{AIR.InformationModel.Reinsurance}AppliesToLineOfBusiness')

    assert og_text != new_tags.text

    return new_filename


# CALL METHOD

#xml_edit('Glatfelter2Q2019.xml')



