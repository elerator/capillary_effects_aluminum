from PyQt5.QtWidgets import QTreeView, QAbstractItemView

class LimitedDropTree(QTreeView):
    #Make sure to set QStandardItemModel item's flags to prohibit pasting into them: Allow only toplevel drop
    #item.setFlags( Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemIsEditable)
    def __init__(self, parent):
        QTreeView.__init__(self, parent)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        #setSelectionMode(QAbstractItemView::SingleSelection);

    def dropEvent(self, evt):
        """ Executes dropEvent when taegetlocation is within first row"""
        if self.header().sectionSize(0)>evt.pos().x():#Only execute drop when on first column
            QTreeView.dropEvent(self, evt)
