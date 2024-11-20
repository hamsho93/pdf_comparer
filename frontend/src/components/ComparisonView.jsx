import React, { useState } from 'react';
import '../styles/ComparisonView.css';

const ComparisonView = ({ oldPdf, newPdf, changes }) => {
  return (
    <div className="comparison-container">
      <div className="pdf-container">
        <div className="pdf-viewer">
          <h3>Original Document</h3>
          <iframe
            src={`${oldPdf}#toolbar=0`}
            title="Original PDF"
            className="pdf-frame"
          />
        </div>
        
        <div className="pdf-viewer">
          <h3>New Document</h3>
          <iframe
            src={`${newPdf}#toolbar=0`}
            title="New PDF"
            className="pdf-frame"
          />
        </div>
      </div>

      <div className="changes-panel">
        <h3>Changes</h3>
        <div className="changes-list">
          {changes.map((change, index) => (
            <div 
              key={index}
              className={`change-item ${change.type.toLowerCase().replace(' ', '-')}`}
            >
              <div className="change-header">
                <span className="change-type">{change.type}</span>
                {change.metadata.new_section && (
                  <span className="change-section">{change.metadata.new_section}</span>
                )}
              </div>
              <div className="change-diff">{change.diff}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ComparisonView;