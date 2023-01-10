from PySide6.QtWidgets import QWidget, QTreeWidget, QTreeWidgetItem

from deepx.ui.components import NetworkItem

class HomeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        tree = QTreeWidget(self)
        tree.setHeaderHidden(True)

        top_item1 = QTreeWidgetItem(tree)
        top_item1.setSelected(True)
        top_item1.setText(0, 'Project1')
        child_item1 = QTreeWidgetItem(top_item1)
        child_item1.setText(0, 'network1')
        child_item2 = QTreeWidgetItem(top_item1)
        child_item2.setText(0, 'network2')

        top_item2 = QTreeWidgetItem(tree)
        top_item2.setText(0, 'Project2')
        child_item1 = QTreeWidgetItem(top_item2)
        child_item1.setText(0, 'network1')
        child_item2 = QTreeWidgetItem(top_item2)
        child_item2.setText(0, 'network2')

        tree.expandAll()
        tree.itemClicked.connect(self.on_tree_click)

    def on_tree_click(self, item:QTreeWidgetItem, index):
        print(item.text(index))
        if item.parent():
            print(item.parent().text(index))
